import { Tabs } from 'expo-router';
import React from 'react';

import { HapticTab } from '@/components/haptic-tab';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

// Kinetra is a single-screen app (Home) -- everything (sessions, calibration,
// team mode, settings) lives inside app/(tabs)/index.tsx as internal view
// state, not as separate tabs. The tab bar is kept structurally (rather than
// ripping out the (tabs) group and Tabs.Screen setup) but hidden, since a
// visible tab bar with only one destination is dead chrome with nothing to
// switch between. If a second real top-level destination is ever added,
// just add its Tabs.Screen back and remove tabBarStyle below.
export default function TabLayout() {
  const colorScheme = useColorScheme();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: Colors[colorScheme ?? 'light'].tint,
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarStyle: { display: 'none' },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="house.fill" color={color} />,
        }}
      />
    </Tabs>
  );
}
