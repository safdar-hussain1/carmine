/**
 * The shade catalogue -- what a physical counter would call the range.
 *
 * Each product gets a named ramp built the way a cosmetics line builds one:
 * ordered from the lightest, most wearable end toward the deepest, so
 * scanning the row left to right is scanning depth rather than hue chaos.
 * Every preset colour in `gen/constants.json` appears somewhere in its
 * product's ramp, so tapping a preset look always lands on a named shade
 * rather than on an unnamed colour the swatch row cannot show as selected.
 */

import type { Finish, ProductName } from "../engine/look";

export interface Shade {
  name: string;
  hex: string;
}

export interface ProductMeta {
  name: ProductName;
  /** Display name in the rail. */
  label: string;
  /** One line describing what the product does to the frame. */
  blurb: string;
  /** Shade ramp, light to deep. */
  shades: Shade[];
  /** Intensity used when the product is switched on from zero. */
  defaultIntensity: number;
  /** Only lipstick branches on finish in the engine. */
  hasFinish?: boolean;
}

const LIPSTICK: Shade[] = [
  { name: "Bare Silk", hex: "#C99A93" },
  { name: "Petal", hex: "#D98E92" },
  { name: "Rosewater", hex: "#C4707F" },
  { name: "Ballet", hex: "#CE7C8E" },
  { name: "Blush Rose", hex: "#BE6B84" },
  { name: "Old Rose", hex: "#B03A5B" },
  { name: "Terracotta", hex: "#B4563C" },
  { name: "Brick Ember", hex: "#9E3B2B" },
  { name: "Spiced Cinnamon", hex: "#8E4A3A" },
  { name: "Cherry Cordial", hex: "#A81E32" },
  { name: "Carmine", hex: "#960018" },
  { name: "Velvet Red", hex: "#8E1B3A" },
  { name: "Mulberry", hex: "#7A2B4E" },
  { name: "Damson", hex: "#6B2340" },
  { name: "Black Cherry", hex: "#55182F" },
  { name: "Cocoa Nude", hex: "#8B5C50" },
];

const EYESHADOW: Shade[] = [
  { name: "Porcelain", hex: "#E8D5CB" },
  { name: "Champagne", hex: "#DCC1A4" },
  { name: "Sand", hex: "#C9A98A" },
  { name: "Antique Gold", hex: "#A98A4B" },
  { name: "Fawn", hex: "#B08968" },
  { name: "Bronze Leaf", hex: "#8C6234" },
  { name: "Toasted Almond", hex: "#8A5A44" },
  { name: "Rosewood", hex: "#8A5560" },
  { name: "Mauve Ash", hex: "#7E6478" },
  { name: "Heather", hex: "#6F5A7E" },
  { name: "Cocoa", hex: "#6E4632" },
  { name: "Aubergine", hex: "#5C3A6E" },
  { name: "Storm", hex: "#4E4F5C" },
  { name: "Soft Smoke", hex: "#3A3438" },
];

const EYELINER: Shade[] = [
  { name: "Warm Taupe", hex: "#6A5648" },
  { name: "Cocoa Brown", hex: "#4E3527" },
  { name: "Espresso", hex: "#3B2A22" },
  { name: "Slate", hex: "#37414A" },
  { name: "Soft Charcoal", hex: "#33312F" },
  { name: "Navy Ink", hex: "#1E2A44" },
  { name: "Deep Teal", hex: "#17383A" },
  { name: "Plum Ink", hex: "#2E1A2C" },
  { name: "Forest Ink", hex: "#1F2E22" },
  { name: "Bordeaux", hex: "#43151F" },
  { name: "Bitter Chocolate", hex: "#2C1D16" },
  { name: "Ink Black", hex: "#1B1B1B" },
];

const BROWS: Shade[] = [
  { name: "Ash Blonde", hex: "#A08A6E" },
  { name: "Honey Blonde", hex: "#9C7A4E" },
  { name: "Taupe", hex: "#8A7460" },
  { name: "Soft Brown", hex: "#6E5744" },
  { name: "Auburn", hex: "#6A3520" },
  { name: "Ash Brown", hex: "#5E4E40" },
  { name: "Chestnut", hex: "#5A3C28" },
  { name: "Warm Brunette", hex: "#4A3728" },
  { name: "Mahogany", hex: "#4B2418" },
  { name: "Cocoa Bark", hex: "#3F2C1E" },
  { name: "Dark Neutral", hex: "#33261C" },
  { name: "Soft Black", hex: "#241C16" },
];

const BLUSH: Shade[] = [
  { name: "Peach Whisper", hex: "#F0A98C" },
  { name: "Apricot", hex: "#E89A72" },
  { name: "Coral Bloom", hex: "#E5806E" },
  { name: "Warm Petal", hex: "#E08A82" },
  { name: "Rose Glow", hex: "#D96C6C" },
  { name: "Dusty Rose", hex: "#C87783" },
  { name: "Terracotta Warmth", hex: "#C4674A" },
  { name: "Antique Pink", hex: "#BC6A78" },
  { name: "Bronze Sunlit", hex: "#B4744F" },
  { name: "Raspberry", hex: "#B24A63" },
  { name: "Mauve Haze", hex: "#A9738A" },
  { name: "Soft Berry", hex: "#A34A5C" },
  { name: "Wine Flush", hex: "#94374E" },
  { name: "Plum Veil", hex: "#8C5670" },
];

const HIGHLIGHTER: Shade[] = [
  { name: "Moonstone", hex: "#F3E7E1" },
  { name: "Candlelight", hex: "#F7E3C0" },
  { name: "Pearl", hex: "#F5D9C8" },
  { name: "Vanilla Gold", hex: "#EFD9A6" },
  { name: "Champagne", hex: "#F0DCB6" },
  { name: "Peach Pearl", hex: "#F5CBAE" },
  { name: "Rose Quartz", hex: "#F2CFC7" },
  { name: "Opal", hex: "#E8DCE6" },
  { name: "Icy Lilac", hex: "#DED4E8" },
  { name: "Soft Gold", hex: "#E7C68B" },
  { name: "Warm Bronze", hex: "#DCB58C" },
  { name: "Copper Glow", hex: "#D9A377" },
];

/**
 * Rail order -- what people reach for first, not the order the shader
 * paints in (`PRODUCT_ORDER`). Lips lead because lipstick is the shade
 * anyone tries first.
 */
export const PRODUCTS: ProductMeta[] = [
  {
    name: "lipstick",
    label: "Lips",
    blurb: "Tinted in Lab so the lip's own texture and highlights survive.",
    shades: LIPSTICK,
    defaultIntensity: 0.6,
    hasFinish: true,
  },
  {
    name: "eyeshadow",
    label: "Eyes",
    blurb: "A soft wash over the lid, feathered up into the crease.",
    shades: EYESHADOW,
    defaultIntensity: 0.4,
  },
  {
    name: "eyeliner",
    label: "Liner",
    blurb: "The one product painted flat: covering the lash line is the point.",
    shades: EYELINER,
    defaultIntensity: 0.7,
  },
  {
    name: "brows",
    label: "Brows",
    blurb: "A low-opacity tint along the brow arch, never a drawn-on shape.",
    shades: BROWS,
    defaultIntensity: 0.35,
  },
  {
    name: "blush",
    label: "Cheeks",
    blurb: "An ellipse on the apple of each cheek, scaled to your face.",
    shades: BLUSH,
    defaultIntensity: 0.3,
  },
  {
    name: "highlighter",
    label: "Glow",
    blurb: "Cheekbone, nose bridge and brow bone, lifted at the highlights.",
    shades: HIGHLIGHTER,
    defaultIntensity: 0.5,
  },
];

export const FINISHES: Array<{ value: Finish; label: string; note: string }> = [
  { value: "matte", label: "Matte", note: "Damps micro-highlights toward a blurred lightness." },
  { value: "satin", label: "Satin", note: "The tint alone: no highlight pass either way." },
  { value: "gloss", label: "Gloss", note: "Lifts the brightest quarter of the lip." },
];

/** Finds the catalogue entry for a hex value, case-insensitively. */
export function shadeName(product: ProductName, hex: string): string | null {
  const meta = PRODUCTS.find((p) => p.name === product);
  if (!meta) {
    return null;
  }
  const wanted = hex.toLowerCase();
  return meta.shades.find((s) => s.hex.toLowerCase() === wanted)?.name ?? null;
}
