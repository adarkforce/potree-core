import type {BufferAttribute} from 'three';
import type {ReleasableAttribute} from '../utils/memory-budget';

/**
 * Attributes whose CPU array stays after upload: positions (CPU picking,
 * selection, the overlay's point counts) and classification (overlays write
 * results into it in place).
 */
export const KEEP_CPU_ATTRIBUTES: ReadonlySet<string> = new Set(['position', 'classification']);

/**
 * Attributes the app asked to keep on the CPU as well (e.g. while a live
 * style filter tests them, so CPU picking can mirror the shader). Applies to
 * nodes decoded from now on; counted per request so callers can nest.
 */
const retainedCpuAttributes = new Map<string, number>();

export function retainCpuAttribute(name: string): void
{
	retainedCpuAttributes.set(name, (retainedCpuAttributes.get(name) ?? 0) + 1);
}

export function releaseCpuAttribute(name: string): void
{
	const n = (retainedCpuAttributes.get(name) ?? 0) - 1;
	if (n > 0)
	{
		retainedCpuAttributes.set(name, n);
	}
	else
	{
		retainedCpuAttributes.delete(name);
	}
}

/** Whether a decoded attribute keeps its CPU array after upload. */
export function keepsCpuArray(name: string): boolean
{
	return KEEP_CPU_ATTRIBUTES.has(name) || retainedCpuAttributes.has(name);
}

/**
 * Potree 2 attribute names (metadata.json) that are not GLSL identifiers,
 * mapped to the names pointcloud.vs binds. Without this, return number,
 * number of returns, point source id and gps time never reach the shader.
 */
export const SHADER_ATTRIBUTE_ALIASES: Readonly<Record<string, string>> = {
	'return number': 'returnNumber',
	'number of returns': 'numberOfReturns',
	'point source id': 'pointSourceID',
	'gps-time': 'gpsTime',
};

type TypedArrayConstructor = new (length: number) => BufferAttribute['array'];

/**
 * Once three.js has uploaded the attribute, replace its CPU array with an
 * empty one of the same type and remember the GPU size (for byte accounting).
 * The GL buffer keeps the data; the attribute can no longer be re-uploaded,
 * so nobody may free its GL buffer while the node lives (see
 * isReleased, checked by the app's overlay injection).
 */
export function releaseArrayOnUpload(attribute: BufferAttribute): void
{
	attribute.onUpload(function(this: BufferAttribute)
	{
		if (this.array.length === 0)
		{
			return;
		}
		(this as BufferAttribute & ReleasableAttribute).gpuByteLength = (this.array as unknown as ArrayBufferView).byteLength;
		this.array = new (this.array.constructor as TypedArrayConstructor)(0);
	});
}

/** Whether an attribute's CPU copy was released after upload. */
export function isReleased(attribute: BufferAttribute): boolean
{
	return (attribute as BufferAttribute & ReleasableAttribute).gpuByteLength !== undefined && attribute.array.length === 0;
}
