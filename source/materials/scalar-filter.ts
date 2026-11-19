import type {ScalarFilter} from './point-cloud-material';

/**
 * True when the scalar range filter hides a point with this overlayScalar
 * value: outside [low, high], or NaN (a NaN lies in no range). The CPU twin of
 * scalarFilterHides() in pointcloud.vs, for picking and other CPU passes that
 * must agree with what is drawn. A null filter hides nothing.
 */
export function scalarFilterHides(value: number, filter: ScalarFilter | null): boolean
{
	return filter !== null && !(value >= filter.low && value <= filter.high);
}
