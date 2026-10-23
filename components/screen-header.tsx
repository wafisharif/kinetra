import { Pressable, StyleSheet, Text } from 'react-native';

type ScreenHeaderProps = {
  onBack: () => void;
  label?: string;
};

// A persistent, top-of-screen back control, reused across every internal
// Kinetra view. Added because every one of these screens previously put its
// only way back at the very bottom of a (sometimes very long) ScrollView --
// this sits at the top instead, right where a user looks first. The
// existing bottom button is left in place everywhere too, so this is purely
// additive: nothing that worked before stops working.
export function ScreenHeader({ onBack, label = 'Home' }: ScreenHeaderProps) {
  return (
    <Pressable
      style={styles.row}
      onPress={onBack}
      accessibilityRole="button"
      accessibilityLabel={`Back to ${label}`}
      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
    >
      <Text style={styles.chevron}>{'‹'}</Text>
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 4,
    paddingVertical: 8,
    paddingHorizontal: 2,
    marginBottom: 14,
  },
  chevron: {
    color: '#93c5fd',
    fontSize: 24,
    fontWeight: '700',
    lineHeight: 24,
  },
  label: {
    color: '#93c5fd',
    fontSize: 16,
    fontWeight: '600',
  },
});
