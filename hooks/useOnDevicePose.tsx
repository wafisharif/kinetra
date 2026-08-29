import { useCallback, useEffect, useRef, useState } from 'react';
import { View } from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';
import type { ComponentType } from 'react';

// ---------------------------------------------------------------------------
// On-device pose hook (Step 1, v3 — WebView worker, inlined).
//
// Why a WebView? @mediapipe/tasks-vision ships as an .mjs ES module that
// uses a dynamic `import(t.toString())` pattern Metro cannot parse. The
// fix is to never let Metro see the MediaPipe source — we hand it to a
// hidden WebView, which loads the same library from a CDN at runtime.
//
// The HTML the WebView runs is inlined below as a data URI. Two reasons:
//   1. Metro's dev server doesn't reliably serve /public files in every
//      setup, which gave us "URL not found" before.
//   2. Inlining means the worker is self-contained — no asset pipeline,
//      no fetch, no race with the dev server's startup.
//
// The hook returns:
//   - ready:    true once the worker has loaded MediaPipe and is ready
//   - error:    last error message from the worker, if any
//   - detect:   send a base64 JPEG, get back a landmark count (0–33)
//   - Worker:   a React component to render once, somewhere in the tree
// ---------------------------------------------------------------------------

// Pinned to a known-working version of MediaPipe Tasks Vision.
const MEDIAPIPE_VERSION = '0.10.18';
const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MEDIAPIPE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// The HTML runs inside an isolated WebView and has no access to Metro.
// It only talks back to React Native via window.ReactNativeWebView.postMessage.
const WORKER_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Pose Worker</title>
    <style>
      html, body {
        margin: 0;
        padding: 0;
        background: transparent;
        color: #fff;
        font: 12px -apple-system, system-ui, sans-serif;
        overflow: hidden;
      }
      #status { padding: 4px 6px; white-space: nowrap; }
    </style>
  </head>
  <body>
    <div id="status">pose: loading…</div>
    <canvas id="stage" width="640" height="480"></canvas>
    <script>
      (async function () {
        var statusEl = document.getElementById('status');
        var canvas = document.getElementById('stage');
        var ctx = canvas.getContext('2d');

        function setStatus(msg) {
          statusEl.textContent = 'pose: ' + msg;
        }
        function post(type, extra) {
          try {
            window.ReactNativeWebView.postMessage(
              JSON.stringify(Object.assign({ type: type }, extra || {}))
            );
          } catch (e) {}
        }
        function loadImage(src) {
          return new Promise(function (resolve, reject) {
            var img = new Image();
            img.onload = function () { resolve(img); };
            img.onerror = function () { reject(new Error('image load failed')); };
            img.src = src;
          });
        }
        async function handleMessage(raw) {
          var msg;
          try { msg = typeof raw === 'string' ? JSON.parse(raw) : raw; }
          catch (e) { return; }
          if (!msg || msg.type !== 'detect') return;
          var requestId = msg.requestId;
          var dataUri = msg.dataUri;
          if (!dataUri) { post('result', { requestId: requestId, count: 0 }); return; }
          try {
            var img = await loadImage(dataUri);
            if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
              canvas.width = img.naturalWidth;
              canvas.height = img.naturalHeight;
            }
            ctx.drawImage(img, 0, 0);
            var result = landmarker.detect(canvas);
            var landmarks = result && result.landmarks && result.landmarks[0];
            post('result', { requestId: requestId, count: landmarks ? landmarks.length : 0 });
          } catch (e) {
            post('error', { message: 'detect failed: ' + (e && e.message ? e.message : String(e)) });
            post('result', { requestId: requestId, count: 0 });
          }
        }

        try {
          setStatus('loading mediapipe…');
          var mod = await import(${JSON.stringify(MEDIAPIPE_URL)});
          var FilesetResolver = mod.FilesetResolver;
          var PoseLandmarker = mod.PoseLandmarker;

          setStatus('loading wasm…');
          var fileset = await FilesetResolver.forVisionTasks(${JSON.stringify(MEDIAPIPE_WASM_URL)});

          setStatus('loading model…');
          var landmarker = await PoseLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: ${JSON.stringify(MEDIAPIPE_MODEL_URL)},
              delegate: 'GPU'
            },
            runningMode: 'IMAGE',
            numPoses: 1
          });

          setStatus('ready');
          post('ready');

          window.addEventListener('message', function (event) { handleMessage(event.data); });
          document.addEventListener('message', function (event) { handleMessage(event.data); });
        } catch (e) {
          var message = e && e.message ? e.message : String(e);
          setStatus('error: ' + message);
          post('error', { message: message });
        }
      })();
    </script>
  </body>
</html>`;

// The worker HTML is inlined as a JS string and handed to the WebView via
// `source={{ html: WORKER_HTML }}` below. We don't load it from Metro's dev
// server (`{ uri: 'pose-worker.html' }`) because Metro doesn't reliably
// serve files from the project root in every setup — when the file fails
// to load, the WebView shows an error page, never runs MediaPipe, and the
// UI gets stuck on "warming up…" forever. Inlining makes the worker
// self-contained: no asset pipeline, no fetch, no race with the dev server.
const WORKER_SOURCE = { html: WORKER_HTML, baseUrl: 'https://localhost' };

type WorkerMessage =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  | { type: 'result'; requestId: number; count: number };

type WorkerPostMessage = (message: string) => void;

type UseOnDevicePoseResult = {
  ready: boolean;
  error: string | null;
  detect: (base64Jpeg: string) => Promise<number>;
  Worker: ComponentType;
};

type WorkerHandle = {
  postMessage: WorkerPostMessage;
};

export function useOnDevicePose(): UseOnDevicePoseResult {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const webViewRef = useRef<WorkerHandle | null>(null);
  // On Android, postMessage() is silently dropped until the WebView has
  // finished loading its source. We track the load state so detect() never
  // fires into a not-yet-ready WebView. On iOS this is essentially instant.
  const webViewLoadedRef = useRef(false);

  const pendingRef = useRef<Map<number, (count: number) => void>>(new Map());
  const nextRequestIdRef = useRef(1);

  const onMessage = useCallback((event: WebViewMessageEvent) => {
    let msg: WorkerMessage | null = null;
    try {
      msg = JSON.parse(event.nativeEvent.data) as WorkerMessage;
    } catch {
      return;
    }
    if (!msg) return;

    if (msg.type === 'ready') {
      setReady(true);
      setError(null);
      return;
    }

    if (msg.type === 'error') {
      setError(msg.message);
      return;
    }

    if (msg.type === 'result') {
      const resolve = pendingRef.current.get(msg.requestId);
      if (resolve) {
        pendingRef.current.delete(msg.requestId);
        resolve(msg.count);
      }
    }
  }, []);

  const detect = useCallback(
    (base64Jpeg: string): Promise<number> => {
      const handle = webViewRef.current;
      if (!handle || !ready || !webViewLoadedRef.current) {
        return Promise.resolve(0);
      }

      const requestId = nextRequestIdRef.current++;
      return new Promise<number>((resolve) => {
        const timeoutId = setTimeout(() => {
          // If the WebView never replies within 5 s, treat it as a missed
          // detection. This prevents the UI from getting stuck at 0/33 when
          // the round-trip silently fails (a real failure mode on Android
          // when the WebView loses focus during recording).
          if (pendingRef.current.delete(requestId)) {
            resolve(0);
          }
        }, 5000);

        pendingRef.current.set(requestId, (count) => {
          clearTimeout(timeoutId);
          resolve(count);
        });

        try {
          handle.postMessage(
            JSON.stringify({
              type: 'detect',
              requestId,
              dataUri: base64Jpeg,
            }),
          );
        } catch (err) {
          // If postMessage itself throws (e.g. WebView torn down), resolve
          // immediately so the snapshot loop can keep ticking.
          if (pendingRef.current.delete(requestId)) {
            clearTimeout(timeoutId);
            resolve(0);
          }
        }
      });
    },
    [ready],
  );

  useEffect(() => {
    return () => {
      pendingRef.current.forEach((resolve) => resolve(0));
      pendingRef.current.clear();
    };
  }, []);

  const Worker = useCallback(() => {
    return (
      // The wrapper exists solely to give the absolutely-positioned
      // WebView its own positioning context. The wrapper itself is zero
      // size so it never claims layout space, and `overflow: hidden`
      // makes sure the WebView can never paint outside its 1x1 box.
      <View
        style={{
          width: 0,
          height: 0,
          position: 'absolute',
          overflow: 'hidden',
        }}
        pointerEvents="none"
      >
        <WebView
          // 1x1 transparent WebView. `pointerEvents="none"` so it never
          // blocks touches. We hand the worker HTML straight to the WebView
          // via `source={{ html: ... }}` so Metro's dev server is not in the
          // loop — see WORKER_SOURCE above for the full reason.
          style={{
            position: 'absolute',
            width: 1,
            height: 1,
            opacity: 0,
            left: 0,
            top: 0,
          }}
          pointerEvents="none"
          // Allow any origin so the worker can reach cdn.jsdelivr.net and
          // storage.googleapis.com for the MediaPipe WASM and model files.
          originWhitelist={['*']}
          source={WORKER_SOURCE}
          onMessage={onMessage}
          onLoad={() => {
            // Mark the WebView as ready to receive postMessage. The actual
            // MediaPipe model still has to download (handled by the worker's
            // own 'ready' message), but this gate is what protects us from
            // Android silently dropping pre-load postMessage calls.
            webViewLoadedRef.current = true;
          }}
          // If the WebView itself fails to render (bad HTML, asset load
          // failure, etc.) surface that to the user instead of leaving the
          // UI stuck on "warming up…". Without these handlers the failure
          // is completely silent, which is exactly the bug that motivated
          // the fix in the first place.
          onError={(event) => {
            const message =
              (event as { nativeEvent?: { description?: string } }).nativeEvent
                ?.description ?? 'WebView failed to load';
            setError(message);
            webViewLoadedRef.current = true;
          }}
          onHttpError={(event) => {
            const status =
              (event as { nativeEvent?: { statusCode?: number } }).nativeEvent
                ?.statusCode;
            if (typeof status === 'number' && status >= 400) {
              setError(`WebView HTTP ${status}`);
            }
            webViewLoadedRef.current = true;
          }}
          ref={(ref: WorkerHandle | null) => {
            webViewRef.current = ref;
          }}
          javaScriptEnabled
          domStorageEnabled
          mixedContentMode="always"
        />
      </View>
    );
  }, [onMessage]);

  return { ready, error, detect, Worker };
}