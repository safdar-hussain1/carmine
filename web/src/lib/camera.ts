/**
 * Camera and photo-upload input sources.
 *
 * `startCamera` only touches `navigator.mediaDevices.getUserMedia` and
 * standard `HTMLVideoElement` playback APIs, so it keeps working when a test
 * harness stubs `getUserMedia` to resolve with `canvas.captureStream(30)`
 * instead of a real camera stream.
 */

export interface FrameSize {
  width: number;
  height: number;
}

const MAX_PHOTO_LONG_SIDE = 1600;

/**
 * Requests the front-facing camera, attaches the resulting stream to
 * `video`, and resolves once the video's intrinsic size is known. Mirroring
 * for display is a CSS concern (transform: scaleX(-1)) handled by the UI
 * layer, not here.
 */
export async function startCamera(video: HTMLVideoElement): Promise<FrameSize> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;

  await new Promise<void>((resolve, reject) => {
    const onLoaded = () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
      reject(new Error("video element failed to load the camera stream"));
    };
    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("error", onError);
  });

  await video.play();

  return { width: video.videoWidth, height: video.videoHeight };
}

/** Stops every track on a stream previously attached by `startCamera`. */
export function stopCamera(video: HTMLVideoElement): void {
  const stream = video.srcObject;
  if (stream instanceof MediaStream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
  video.srcObject = null;
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to decode image file"));
    img.src = url;
  });
}

/**
 * Decodes an uploaded image file into a canvas, downscaling so its long
 * side is at most `MAX_PHOTO_LONG_SIDE` pixels. EXIF orientation is not
 * read -- the canvas is drawn exactly as the browser decodes the file.
 */
export async function loadPhoto(file: File): Promise<HTMLCanvasElement> {
  const objectUrl = URL.createObjectURL(file);
  let img: HTMLImageElement;
  try {
    img = await loadImage(objectUrl);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }

  const longSide = Math.max(img.naturalWidth, img.naturalHeight);
  const scale = longSide > MAX_PHOTO_LONG_SIDE ? MAX_PHOTO_LONG_SIDE / longSide : 1;
  const width = Math.round(img.naturalWidth * scale);
  const height = Math.round(img.naturalHeight * scale);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  ctx.drawImage(img, 0, 0, width, height);
  return canvas;
}
