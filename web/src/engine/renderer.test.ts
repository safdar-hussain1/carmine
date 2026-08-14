import { describe, expect, it } from "vitest";

import { createRenderer } from "./renderer";

/**
 * The shader itself is compiled and exercised by the in-browser selftest --
 * there is no WebGL2 context in the Node test runner, and a string-level
 * check of GLSL source would prove nothing a driver would agree with.
 *
 * What is worth testing here is the contract the caller depends on when
 * there is no GPU path available: `createRenderer` reports that by
 * returning null, not by throwing, because an old browser reaching this
 * function is an expected outcome that the caller handles by falling back.
 */
describe("createRenderer", () => {
  it("returns null instead of throwing when WebGL2 is unavailable", () => {
    const canvas = { getContext: () => null } as unknown as HTMLCanvasElement;
    expect(createRenderer(canvas)).toBeNull();
  });

  it("asks for a WebGL2 context specifically", () => {
    const requested: string[] = [];
    const canvas = {
      getContext(kind: string) {
        requested.push(kind);
        return null;
      },
    } as unknown as HTMLCanvasElement;
    createRenderer(canvas);
    expect(requested).toEqual(["webgl2"]);
  });
});
