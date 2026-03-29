/**
 * Enumerates the types of workers available in the worker pool.
 */
export enum WorkerType {
	/**
	 * Worker for decoding Brotli-compressed data.
	 */
	DECODER_WORKER_BROTLI = 'DECODER_WORKER_BROTLI',

	/**
	 * Worker for general decoding tasks.
	 */
	DECODER_WORKER = 'DECODER_WORKER',
}

/**
 * Creates a new worker instance based on the specified worker type.
 * Uses Vite-compatible `new Worker(new URL(...), ...)` pattern.
 */
function createWorker(type: WorkerType): Worker
{
	switch (type)
	{
		case WorkerType.DECODER_WORKER_BROTLI: {
			return new Worker(new URL('./brotli-decoder.worker.js', import.meta.url), { type: 'module' });
		}
		case WorkerType.DECODER_WORKER: {
			return new Worker(new URL('./decoder.worker.js', import.meta.url), { type: 'module' });
		}
		default:
			throw new Error('Unknown worker type');
	}
}

/**
 * WorkerPool manages a collection of worker instances, allowing for efficient retrieval and return of workers based on their type.
 */
export class WorkerPool
{
	/**
	 * Workers will be an object that has a key for each worker type and the value is an array of Workers that can be empty.
	 */
	public workers: { [key in WorkerType]: Worker[] } = {DECODER_WORKER: [], DECODER_WORKER_BROTLI: []};

	/**
	 * Retrieves a Worker instance from the pool associated with the specified worker type.
	 */
	public getWorker(workerType: WorkerType): Worker
	{
		if (this.workers[workerType] === undefined)
		{
			throw new Error('Unknown worker type');
		}
		if (this.workers[workerType].length === 0)
		{
			let worker = createWorker(workerType);
			this.workers[workerType].push(worker);
		}
		let worker = this.workers[workerType].pop();
		if (worker === undefined)
		{
			throw new Error('No workers available');
		}
		return worker;
	}

	/**
	 * Returns a worker instance to the pool for the specified worker type.
	 */
	public returnWorker(workerType: WorkerType, worker: Worker)
	{
		this.workers[workerType].push(worker);
	}
}
