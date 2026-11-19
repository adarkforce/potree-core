import {LOAD_RETRY_BASE_MS} from '../constants';

/**
 * Delay before the next attempt after `attempts` failed loads of a node:
 * LOAD_RETRY_BASE_MS, then four times longer per further failure.
 */
export function loadRetryDelayMs(attempts: number): number
{
	if (attempts <= 0)
	{
		return 0;
	}
	return LOAD_RETRY_BASE_MS * 4 ** (attempts - 1);
}
