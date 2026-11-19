/**
 * Live style: the uniform-only effects of pointcloud.vs (class visibility in
 * every colour mode, attribute range filters, a cloud-frame clip box, the
 * scalar source with covariance features / slope / verticality, a slope
 * filter over stored normals, and per-node result states).
 *
 * Every value here is a uniform: changing it never recompiles the shader,
 * uploads a buffer or posts to a worker. The hide tests use only exactly
 * rounded float32 operations so a CPU twin can agree with the GPU bit for bit.
 */

/** What the scalar colour mode and the scalar range filter read. */
export enum ScalarSource {
	OVERLAY = 0,
	FEATURE = 1,
	SLOPE = 2,
	VERTICALITY = 3,
}

/** Covariance features, in the WASM ops' ScalarFieldId order (pointcloud-ops-wasm Types.hpp). */
export enum CovarianceFeature {
	LINEARITY = 0,
	PLANARITY = 1,
	SPHERICITY = 2,
	OMNIVARIANCE = 3,
	ANISOTROPY = 4,
	EIGENENTROPY = 5,
	SURFACE_VARIATION = 6,
	VERTICALITY = 7,
}

/** Raw attributes an attribute range filter can test (values in the attribute's stored units). */
export enum FilterAttribute {
	OFF = 0,
	INTENSITY = 1,
	RETURN_NUMBER = 2,
	NUMBER_OF_RETURNS = 3,
	POINT_SOURCE_ID = 4,
	GPS_TIME = 5,
	CLASSIFICATION = 6,
}

/** Geometry attribute each FilterAttribute reads (after OctreeLoader's aliasing). */
export const FILTER_ATTRIBUTE_NAMES: Readonly<Record<FilterAttribute, string | null>> = {
	[FilterAttribute.OFF]: null,
	[FilterAttribute.INTENSITY]: 'intensity',
	[FilterAttribute.RETURN_NUMBER]: 'returnNumber',
	[FilterAttribute.NUMBER_OF_RETURNS]: 'numberOfReturns',
	[FilterAttribute.POINT_SOURCE_ID]: 'pointSourceID',
	[FilterAttribute.GPS_TIME]: 'gpsTime',
	[FilterAttribute.CLASSIFICATION]: 'classification',
};

export enum ClipAabbMode {
	OFF = 0,
	KEEP_INSIDE = 1,
	KEEP_OUTSIDE = 2,
}

/** Per-node state of a live (view-driven) result. UNTRACKED draws as before. */
export enum NodeResultState {
	UNTRACKED = 0,
	PENDING = 1,
	PREVIEW = 2,
	EXACT = 3,
}

export type NodeResultEntry = Readonly<{state: NodeResultState; since: number}>;

export const MAX_ATTRIBUTE_FILTERS = 4;

/** Ratio features are tested as lo * den <= num <= hi * den. */
const RATIO_FEATURES: ReadonlySet<number> = new Set([
	CovarianceFeature.LINEARITY,
	CovarianceFeature.PLANARITY,
	CovarianceFeature.SPHERICITY,
	CovarianceFeature.ANISOTROPY,
	CovarianceFeature.SURFACE_VARIATION,
]);

export function isRatioFeature(feature: number): boolean
{
	return RATIO_FEATURES.has(feature);
}

/**
 * The scalar filter's [low, high] (display units) mapped to the domain its
 * exact test compares in: the ratio bounds for ratio features, cubes for
 * omnivariance (cbrt is monotonic), |n.z| bounds for slope in degrees. Pure;
 * the GPU and the CPU twin both read the float32 result.
 */
export function scalarFilterDomain(source: ScalarSource, feature: number, low: number, high: number): [number, number]
{
	if (source === ScalarSource.FEATURE && feature === CovarianceFeature.OMNIVARIANCE)
	{
		return [low * low * low, high * high * high];
	}
	if (source === ScalarSource.SLOPE)
	{
		return slopeToAbsNz(low, high);
	}
	return [low, high];
}

/** Slope range in degrees -> |n.z| range (slope = acos |n.z|); outside [0, 90] is open-ended. */
export function slopeToAbsNz(lowDeg: number, highDeg: number): [number, number]
{
	if (Number.isNaN(lowDeg) || Number.isNaN(highDeg))
	{
		return [NaN, NaN];
	}
	const toRad = Math.PI / 180;
	const lo = highDeg >= 90 ? -Infinity : Math.cos(Math.max(highDeg, 0) * toRad);
	const hi = lowDeg <= 0 ? Infinity : Math.cos(Math.min(lowDeg, 90) * toRad);
	return [lo, hi];
}

/** Sets or clears one class bit in a 256-bit mask held as 8 int32 words. */
export function setClassBit(mask: Int32Array, cls: number, hidden: boolean): void
{
	const c = Math.min(255, Math.max(0, Math.trunc(cls)));
	const bit = 1 << (c & 31);
	mask[c >> 5] = hidden ? mask[c >> 5] | bit : mask[c >> 5] & ~bit;
}

export function classBitSet(mask: Int32Array, cls: number): boolean
{
	// As the shader: int(classification + 0.5) in float32, clamped to [0, 255].
	const c = Math.min(255, Math.max(0, Math.trunc(Math.fround(cls + 0.5))));
	return ((mask[c >> 5] >> (c & 31)) & 1) !== 0;
}
