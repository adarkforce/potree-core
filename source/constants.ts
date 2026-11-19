import { Color, Vector4 } from 'three';

export const DEFAULT_RGB_BRIGHTNESS = 0;
export const DEFAULT_RGB_CONTRAST = 0;
export const DEFAULT_RGB_GAMMA = 1;
export const DEFAULT_MAX_POINT_SIZE = 50;
export const DEFAULT_MIN_NODE_PIXEL_SIZE = 50;
export const DEFAULT_MIN_POINT_SIZE = 2;
export const DEFAULT_PICK_WINDOW_SIZE = 15;
export const DEFAULT_POINT_BUDGET = 1_000_000;
export const MAX_LOADS_TO_GPU = 2;
/**
 * Concurrent node loads (each one a fetch plus a decoder worker): all cores
 * but two (the main thread and the compositor), between 4 and 16.
 */
const HARDWARE_THREADS = typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 4;
export const MAX_NUM_NODES_LOADING = Math.min(16, Math.max(4, HARDWARE_THREADS - 2));
/** A node whose load or decode failed this many times is marked failed and skipped. */
export const MAX_NODE_LOAD_ATTEMPTS = 3;
export const PERSPECTIVE_CAMERA = 'PerspectiveCamera';
export const ORTHOGRAPHIC_CAMERA = 'OrthographicCamera';
export const COLOR_BLACK = new Color(0, 0, 0);
export const DEFAULT_HIGHLIGHT_COLOR = new Vector4(1, 0, 0, 1);
