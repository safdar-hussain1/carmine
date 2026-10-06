/**
 * Inline SVG icons, drawn on a 24-unit grid with `currentColor` strokes so
 * they take the colour of whatever they sit in. Small enough to keep here
 * rather than ship an icon font: a font file would be one more download for
 * a handful of glyphs, and these stay sharp at any size.
 */

const stroke = 'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"';

function svg(body: string): string {
  return `<svg viewBox="0 0 24 24" aria-hidden="true" ${stroke}>${body}</svg>`;
}

export const ICONS = {
  camera: svg('<path d="M3 8.5A2 2 0 0 1 5 6.5h2l1.3-2h7.4L17 6.5h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><circle cx="12" cy="13" r="3.6"/>'),
  photo: svg('<rect x="3" y="4.5" width="18" height="15" rx="2"/><circle cx="8.5" cy="10" r="1.6"/><path d="m4 17 5-4.5 4 3.5 3-2.5 4 3.5"/>'),
  download: svg('<path d="M12 4v10m0 0 4-4m-4 4-4-4"/><path d="M4.5 16.5v1.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-1.5"/>'),
  compare: svg('<path d="M12 3.5v17"/><path d="M8 8 4.5 12 8 16"/><path d="m16 8 3.5 4-3.5 4"/>'),
  back: svg('<path d="M9.5 6.5 4 12l5.5 5.5"/><path d="M4.5 12H15a5 5 0 0 1 0 10h-1"/>'),
  steady: svg('<path d="M3 12c2.5-4 5.5-4 9 0s6.5 4 9 0"/><path d="M3 17c2.5-2 5.5-2 9 0s6.5 2 9 0" opacity=".45"/>'),
  play: svg('<path d="M8 5.5v13l10.5-6.5Z"/>'),
  sliders: svg('<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>'),
  close: svg('<path d="M6 6l12 12M18 6 6 18"/>'),
  sun: svg('<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4"/>'),
  moon: svg('<path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.5 8.5 0 1 0 10.2 10.2Z"/>'),
  grip: svg('<path d="M9.5 8 6 12l3.5 4"/><path d="m14.5 8 3.5 4-3.5 4"/>'),
  device: svg('<rect x="4" y="5" width="16" height="11" rx="1.6"/><path d="M2.5 19h19"/><path d="M9.5 10.5l1.8 1.8 3.4-3.6"/>'),
  noUpload: svg('<path d="M12 15V5m0 0-3.5 3.5M12 5l3.5 3.5"/><path d="M5 19h14"/><path d="M4 4l16 16"/>'),
  wifiOff: svg('<path d="M2.5 9a14 14 0 0 1 4.6-2.8M10.4 5.6A14 14 0 0 1 21.5 9"/><path d="M5.5 12.5a9 9 0 0 1 3.6-2M15.6 11.2a9 9 0 0 1 2.9 1.3"/><path d="M9 16a4.5 4.5 0 0 1 6 0"/><circle cx="12" cy="19.2" r=".6" fill="currentColor"/><path d="M3.5 3.5l17 17"/>'),
  github:
    '<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.28 1.19-3.08-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.08 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>',
} as const;
