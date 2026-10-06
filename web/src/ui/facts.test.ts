import { describe, expect, it } from "vitest";

import { FACTS, oneDecimal } from "./facts";

/**
 * The page quotes a handful of measured figures. They are read from the
 * committed reports at build time rather than typed into the copy, and this
 * pins what they come out as -- the same values `tests/test_docs_numbers.py`
 * pins in the README -- so a regenerated report that moves one fails here
 * instead of quietly changing what the page claims.
 */
describe("published facts", () => {
  it("counts the landmarks the model returns", () => {
    expect(FACTS.landmarks).toBe(478);
  });

  it("quotes the lip lightness shift of a flat fill and of carmine", () => {
    expect(FACTS.portraits).toBe(26);
    expect(oneDecimal(FACTS.lipShiftFlat)).toBe("45.7");
    expect(oneDecimal(FACTS.lipShiftCarmine)).toBe("13.2");
  });

  it("quotes the live frame cost and the frame rate it implies", () => {
    expect(oneDecimal(FACTS.frameMs)).toBe("26.6");
    expect(oneDecimal(FACTS.fps)).toBe("37.6");
  });

  it("names the machine the frame cost was measured on", () => {
    expect(FACTS.gpu).toBe("Apple M1 Pro");
  });
});
