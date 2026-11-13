import type { CSSProperties } from 'react';
const paths = {
  book: 'M4 3h10l5 5v13H4z M14 3v6h5 M8 13h7 M8 17h5',
  grid: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
  chat: 'M4 4h16v12H9l-5 4z M8 8h8 M8 12h5',
  search: 'M10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14 M15 15l6 6',
  upload: 'M12 16V3 M7 8l5-5 5 5 M4 15v6h16v-6',
  arrow: 'M4 12h16 M14 6l6 6-6 6',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12 M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
  download: 'M12 3v13 M7 11l5 5 5-5 M4 18v3h16v-3',
  trash: 'M3 6h18 M9 6V3h6v3 M6 6l1 15h10l1-15 M10 10v7 M14 10v7',
  bolt: 'M13 2L4 14h7l-1 8 10-13h-7z',
  close: 'M6 6l12 12 M6 18L18 6',
  lock: 'M6 10h12v11H6z M8 10V6a4 4 0 0 1 8 0v4',
  exit: 'M9 3H3v18h6 M10 12h11 M16 7l5 5-5 5',
  check: 'M5 12l4 4L19 6',
};
export type IconName = keyof typeof paths;
export function Icon({ name, size = 20, style }: { name: IconName; size?: number; style?: CSSProperties }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}><path d={paths[name]} /></svg>;
}
