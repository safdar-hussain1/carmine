/**
 * Inline SVG icons, drawn on a 24-unit grid with `currentColor` strokes so
 * they inherit the theme. Small enough to keep here rather than ship an
 * icon font: a font file would be a second network request, and the page
 * makes none.
 */

const stroke = 'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"';

function svg(body: string): string {
  return `<svg viewBox="0 0 24 24" aria-hidden="true" ${stroke}>${body}</svg>`;
}

export const ICONS = {
  camera: svg('<path d="M3 8.5A2 2 0 0 1 5 6.5h2l1.3-2h7.4L17 6.5h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><circle cx="12" cy="13" r="3.6"/>'),
  photo: svg('<rect x="3" y="4.5" width="18" height="15" rx="2"/><circle cx="8.5" cy="10" r="1.6"/><path d="m4 17 5-4.5 4 3.5 3-2.5 4 3.5"/>'),
  sample: svg('<circle cx="12" cy="12" r="8.5"/><path d="M9 10.5h.01M15 10.5h.01M9 14.5c1.8 1.6 4.2 1.6 6 0"/>'),
  download: svg('<path d="M12 4v10m0 0 4-4m-4 4-4-4"/><path d="M4.5 16.5v1.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-1.5"/>'),
  compare: svg('<path d="M12 3.5v17"/><path d="M8 8 4.5 12 8 16"/><path d="m16 8 3.5 4-3.5 4"/>'),
  stop: svg('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
  sun: svg('<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4"/>'),
  moon: svg('<path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.5 8.5 0 1 0 10.2 10.2Z"/>'),
  grip: svg('<path d="M9.5 8 6 12l3.5 4"/><path d="m14.5 8 3.5 4-3.5 4"/>'),
} as const;
