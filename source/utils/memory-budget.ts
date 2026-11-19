import type {BufferAttribute, BufferGeometry, InterleavedBufferAttribute} from 'three';

/**
 * Byte accounting for resident octree nodes (CPU arrays + GPU buffers) and
 * the byte budget the LRU evicts to.
 */

/** Bytes per point assumed before any node has been measured. */
export const DEFAULT_BYTES_PER_POINT = 100;

/** Share of the device's RAM (navigator.deviceMemory) the viewer may hold in node geometry. */
export const DEVICE_MEMORY_SHARE = 0.25;

/** Resident set above which the LRU starts evicting, as a multiple of the byte budget. */
export const EVICT_HIGH_WATER = 1.2;

/** Recently visible nodes kept beyond the visible set, as a multiple of the point budget's bytes. */
export const CACHE_FACTOR = 1.5;

/** deviceMemory when the browser does not report it (Firefox, Safari). */
const FALLBACK_DEVICE_GB = 4;

/** GPU byte size recorded on an attribute whose CPU array was released after upload. */
export type ReleasableAttribute = (BufferAttribute | InterleavedBufferAttribute) & {gpuByteLength?: number};

export type ResidentBytes = Readonly<{cpu: number; gpu: number}>;

/**
 * CPU bytes still held by a geometry's attribute arrays, and GPU bytes of its
 * buffers (uploaded, or about to be on first draw).
 */
export function geometryBytes(geometry: BufferGeometry | null | undefined): ResidentBytes
{
	if (!geometry)
	{
		return {cpu: 0, gpu: 0};
	}
	let cpu = 0;
	let gpu = 0;
	const attributes = geometry.attributes as Record<string, ReleasableAttribute>;
	for (const name in attributes)
	{
		const attribute = attributes[name];
		const array = (attribute as BufferAttribute).array as ArrayLike<number> & {byteLength?: number} | null;
		const held = array?.byteLength ?? 0;
		cpu += held;
		gpu += attribute.gpuByteLength ?? held;
	}
	return {cpu, gpu};
}

/** The device-memory cap: a share of navigator.deviceMemory (GB, Chrome reports at most 8). */
export function deviceByteCap(deviceMemoryGB: number | undefined): number
{
	const gb = deviceMemoryGB !== undefined && deviceMemoryGB > 0 ? deviceMemoryGB : FALLBACK_DEVICE_GB;
	return Math.floor(gb * 1024 ** 3 * DEVICE_MEMORY_SHARE);
}

/**
 * The byte budget of resident nodes: the point budget's bytes plus a cache of
 * recently visible nodes, never more than the device cap.
 */
export function deriveByteBudget(pointBudget: number, bytesPerPoint: number, deviceCap: number): number
{
	const bpp = bytesPerPoint > 0 && Number.isFinite(bytesPerPoint) ? bytesPerPoint : DEFAULT_BYTES_PER_POINT;
	return Math.min(deviceCap, Math.ceil(pointBudget * bpp * CACHE_FACTOR));
}

/**
 * Points that may be visible at once: the visible nodes (which the LRU never
 * evicts) take at most 1 / CACHE_FACTOR of the byte budget, leaving room for
 * nodes in flight and nodes heavier than the mean. Equals the point budget
 * unless the budget was capped by the device or set explicitly.
 */
export function effectivePointBudget(pointBudget: number, byteBudget: number, bytesPerPoint: number): number
{
	const bpp = bytesPerPoint > 0 && Number.isFinite(bytesPerPoint) ? bytesPerPoint : DEFAULT_BYTES_PER_POINT;
	return Math.max(1, Math.min(pointBudget, Math.floor(byteBudget / (bpp * CACHE_FACTOR))));
}

/** navigator.deviceMemory where the browser exposes it. */
export function navigatorDeviceMemory(): number | undefined
{
	if (typeof navigator === 'undefined')
	{
		return undefined;
	}
	const value = (navigator as Navigator & {deviceMemory?: number}).deviceMemory;
	return typeof value === 'number' ? value : undefined;
}
