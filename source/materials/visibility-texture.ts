/**
 * The visible-nodes texture: one RGBA texel per visible node (child mask,
 * 16-bit offset to the first child, name length), read by the vertex shader
 * for adaptive point sizes and LOD colours.
 *
 * It used to be a fixed 2048×1 texture, so more than 2048 visible nodes threw
 * a RangeError. It is now VISIBLE_NODES_TEXTURE_WIDTH texels wide and as many
 * rows as needed (a power of two, so it is reallocated rarely); the shader
 * addresses it with texelFetch.
 */

export const VISIBLE_NODES_TEXTURE_WIDTH = 2048;

/** Largest parent-to-child offset the 16-bit encoding holds. */
export const MAX_CHILD_OFFSET = 0xffff;

export type VisibleNodeLike = Readonly<{name: string; index: number}>;

/** Rows needed for `count` visible nodes (a power of two, at least 1). */
export function visibilityTextureRows(count: number): number
{
	const rows = Math.max(1, Math.ceil(count / VISIBLE_NODES_TEXTURE_WIDTH));
	return 2 ** Math.ceil(Math.log2(rows));
}

/**
 * Writes the texels of `nodes` (sorted by level, then index: parents before
 * children) into `data` (RGBA bytes, at least nodes.length × 4 long) and the
 * texel of every node into `offsets`. Clears the texels it uses.
 */
export function packVisibleNodes(nodes: readonly VisibleNodeLike[], data: Uint8Array, offsets: Map<string, number>): void
{
	if (data.length < nodes.length * 4)
	{
		throw new RangeError(`visible-nodes texture holds ${data.length / 4} nodes, ${nodes.length} given`);
	}
	data.fill(0, 0, nodes.length * 4);
	offsets.clear();

	for (let i = 0; i < nodes.length; i++)
	{
		const node = nodes[i];
		offsets.set(node.name, i);

		if (i > 0)
		{
			const parentOffset = offsets.get(node.name.slice(0, -1));
			if (parentOffset !== undefined)
			{
				const texel = parentOffset * 4;
				// The first child after the parent is the nearest: keep the smallest offset.
				const current = data[texel + 1] << 8 | data[texel + 2];
				const offset = Math.min(i - parentOffset, MAX_CHILD_OFFSET);
				const toChild = current === 0 ? offset : Math.min(current, offset);

				data[texel] = data[texel] | 1 << node.index;
				data[texel + 1] = toChild >> 8;
				data[texel + 2] = toChild & 0xff;
			}
		}

		data[i * 4 + 3] = node.name.length;
	}
}
