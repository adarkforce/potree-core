import type {BufferAttribute} from 'three';
import type {ReleasableAttribute} from '../utils/memory-budget';

/**
 * Attributes whose CPU array stays after upload: positions (CPU picking,
 * selection, the overlay's point counts) and classification (overlays write
 * results into it in place).
 */
export const KEEP_CPU_ATTRIBUTES: ReadonlySet<string> = new Set(['position', 'classification']);

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
