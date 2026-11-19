import {IPointCloudTreeNode} from '../types';
import {DEFAULT_BYTES_PER_POINT, EVICT_HIGH_WATER} from './memory-budget';

export type Node = IPointCloudTreeNode & {
	/** CPU + GPU bytes the node holds now (OctreeGeometryNode); numPoints × an estimate otherwise. */
	readonly residentBytes?: number;
	readonly parent?: unknown;
};

export class LRUItem
{
	next: LRUItem | null = null;

	previous: LRUItem | null = null;

	/** Bytes accounted for this node (refreshed on every touch). */
	bytes = 0;

	/** Frame of the last touch: nodes touched in the current frame are never evicted. */
	frame = -1;

	constructor(public node: Node) {}
}

/**
 * A doubly-linked-list of the least recently used nodes, bounded in bytes.
 *
 * Each node's bytes (CPU arrays still held + GPU buffers) are re-read when it
 * is touched. Once the resident set exceeds EVICT_HIGH_WATER × byteBudget, the
 * least recently used subtrees are disposed until it is back under the
 * budget. Nodes touched in the current frame (on screen, or just loaded) and
 * the root are never evicted.
 */
export class LRU
{
	/** The least recently used item. */
	public first: LRUItem | null = null;

	/** The most recently used item. */
	public last: LRUItem | null = null;

	/** The total number of points in the LRU cache. */
	public numPoints: number = 0;

	/** CPU + GPU bytes of the nodes in the cache. */
	public numBytes: number = 0;

	/** Nodes disposed by eviction since creation. */
	public evictions: number = 0;

	/** Bytes the resident set is evicted down to (Potree derives it every frame). */
	public byteBudget: number = Infinity;

	/** Items in the LRU cache, indexed by node ID. */
	private items = new Map<number, LRUItem>();

	private frame = 0;

	/**
	 * @param pointBudget The visible point budget (kept for API compatibility; eviction is by bytes).
	 */
	public constructor(public pointBudget: number = 1_000_000) {}

	public get size(): number
	{
		return this.items.size;
	}

	/** Mean bytes per resident point, or undefined while the cache is empty. */
	public get bytesPerPoint(): number | undefined
	{
		return this.numPoints > 0 ? this.numBytes / this.numPoints : undefined;
	}

	/** Starts a new frame: nodes touched before it become evictable. */
	public beginFrame(): void
	{
		this.frame++;
	}

	/**
	 * Checks if the LRU cache contains the specified node.
	 */
	public has(node: Node): boolean
	{
		return this.items.has(node.id);
	}

	/**
	 * Makes the node the most recently used item (adding it if needed) and
	 * re-reads its bytes.
	 */
	public touch(node: Node)
	{
		if (!node.loaded)
		{
			return;
		}

		let item = this.items.get(node.id);
		if (item)
		{
			this.unlink(item);
		}
		else
		{
			item = new LRUItem(node);
			this.items.set(node.id, item);
			this.numPoints += node.numPoints;
		}
		this.append(item);
		item.frame = this.frame;
		const bytes = nodeBytes(node);
		this.numBytes += bytes - item.bytes;
		item.bytes = bytes;
	}

	/**
	 * Removes a node from the LRU cache.
	 */
	public remove(node: Node)
	{
		const item = this.items.get(node.id);
		if (!item)
		{
			return;
		}
		this.unlink(item);
		this.items.delete(node.id);
		this.numPoints -= node.numPoints;
		this.numBytes -= item.bytes;
	}

	/** Removes every node (a cloud was closed). */
	public clear(): void
	{
		this.items.clear();
		this.first = null;
		this.last = null;
		this.numPoints = 0;
		this.numBytes = 0;
	}

	/**
	 * Gets the least recently used item from the LRU cache.
	 */
	public getLRUItem(): Node | undefined
	{
		return this.first ? this.first.node : undefined;
	}

	/**
	 * Above EVICT_HIGH_WATER × byteBudget, disposes least recently used subtrees
	 * until the cache is within the budget or only protected nodes are left.
	 */
	public freeMemory(): void
	{
		if (this.numBytes <= this.byteBudget * EVICT_HIGH_WATER)
		{
			return;
		}
		let cursor = this.first;
		while (cursor && this.numBytes > this.byteBudget)
		{
			if (cursor.frame === this.frame)
			{
				// Everything after it was touched this frame too.
				return;
			}
			if (isRoot(cursor.node))
			{
				cursor = cursor.next;
				continue;
			}
			this.disposeSubtree(cursor.node);
			// The subtree's items left the list: restart from the oldest.
			cursor = this.first;
		}
	}

	/**
	 * Disposes of a subtree starting from the specified node and removes it from the LRU cache.
	 */
	public disposeSubtree(node: Node): void
	{
		const nodesToDispose: Node[] = [];
		node.traverse((n) =>
		{
			if (n.loaded)
			{
				nodesToDispose.push(n as Node);
			}
		});
		if (!nodesToDispose.includes(node))
		{
			nodesToDispose.push(node);
		}

		for (const n of nodesToDispose)
		{
			this.remove(n);
			n.dispose();
			this.evictions++;
		}
	}

	private unlink(item: LRUItem): void
	{
		if (item.previous)
		{
			item.previous.next = item.next;
		}
		else if (this.first === item)
		{
			this.first = item.next;
		}
		if (item.next)
		{
			item.next.previous = item.previous;
		}
		else if (this.last === item)
		{
			this.last = item.previous;
		}
		item.previous = null;
		item.next = null;
	}

	private append(item: LRUItem): void
	{
		item.previous = this.last;
		item.next = null;
		if (this.last)
		{
			this.last.next = item;
		}
		this.last = item;
		if (!this.first)
		{
			this.first = item;
		}
	}
}

function nodeBytes(node: Node): number
{
	const bytes = node.residentBytes;
	return typeof bytes === 'number' && Number.isFinite(bytes) ? bytes : node.numPoints * DEFAULT_BYTES_PER_POINT;
}

/** The root holds the whole cloud's coarsest level and its tree node: it stays. */
function isRoot(node: Node): boolean
{
	return node.parent === null || node.parent === undefined;
}
