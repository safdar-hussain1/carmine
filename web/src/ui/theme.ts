/**
 * Theme: system by default, overridable, remembered.
 *
 * The stored preference is applied by a tiny inline script in index.html
 * before the first paint -- doing it here would be one module-load too late
 * and would show a flash of the wrong ground colour. This module only owns
 * the toggle from then on.
 */

export type ThemePreference = "light" | "dark" | "system";

export const THEME_KEY = "carmine.theme";

function stored(): ThemePreference {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    // Private-mode Safari throws on localStorage access; the toggle still
    // works for the session, it just does not survive a reload.
    return "system";
  }
}

function systemPrefersDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

function resolve(preference: ThemePreference): "light" | "dark" {
  return preference === "system" ? (systemPrefersDark() ? "dark" : "light") : preference;
}

function apply(preference: ThemePreference): void {
  const root = document.documentElement;
  if (preference === "system") {
    delete root.dataset.theme;
  } else {
    root.dataset.theme = preference;
  }
}

/** Wires a toggle button to the theme preference. Returns the current one. */
export function initTheme(button: HTMLButtonElement): ThemePreference {
  let preference = stored();

  const sync = () => {
    const resolved = resolve(preference);
    button.dataset.resolved = resolved;
    button.setAttribute(
      "aria-label",
      resolved === "dark" ? "Switch to the light theme" : "Switch to the dark theme",
    );
  };

  apply(preference);
  sync();

  button.addEventListener("click", () => {
    preference = resolve(preference) === "dark" ? "light" : "dark";
    apply(preference);
    try {
      localStorage.setItem(THEME_KEY, preference);
    } catch {
      // Nothing to do: the theme is applied either way.
    }
  });
  button.addEventListener("click", sync);

  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener("change", sync);

  return preference;
}
