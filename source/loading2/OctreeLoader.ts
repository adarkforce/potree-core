import {BufferAttribute, BufferGeometry, Vector3} from 'three';
import {PointAttribute, PointAttributes, PointAttributeTypes} from './PointAttributes';
import {Box3, Sphere} from 'three';
import {WorkerPool, WorkerType} from './WorkerPool';
import {OctreeGeometryNode} from './OctreeGeometryNode';
import {OctreeGeometry} from './OctreeGeometry';
import {RequestManager} from './RequestManager';
import {MAX_NODE_LOAD_ATTEMPTS} from '../constants';
import {loadRetryDelayMs} from './load-backoff';
import {releaseArrayOnUpload, KEEP_CPU_ATTRIBUTES} from './release-on-upload';

/**
 * Callback invoked after a node's geometry is decoded and ready.
 * Use to inject overlay attributes onto loaded nodes.
 */
export type NodeLoadedCallback = (node: OctreeGeometryNode, geometry: BufferGeometry) => void;

/**
 * NodeLoader is responsible for loading the geometry of octree nodes.
 */
export class NodeLoader
{
	/**
	 * Optional callback invoked after each node geometry is ready.
	 */
	public static onNodeLoaded: NodeLoadedCallback | null = null;

	/** Running totals across every cloud: nodes decoded, loads or decodes that failed. */
	public static readonly stats = {loaded: 0, failures: 0};

	/** Test hook: the next N decodes throw inside the decoder worker. */
	public static debugFailNextDecodes = 0;

	/**
	 * Point attributes to be used when loading the geometry.
	 */
	public attributes?: PointAttributes;

	/**
	 * Scale applied to the geometry when loading.
	 */
	public scale?: [number, number, number];

	/**
	 * Offset applied to the geometry when loading.
	 */
	public offset?: [number, number, number];
	

	constructor(public url: string, public workerPool: WorkerPool, public metadata: Metadata, public requestManager: RequestManager){}

	/**
	 * Loads the geometry for a given octree node.
	 * 
	 * @param node - The octree node to load. 
	 */
	async load(node: OctreeGeometryNode)
	{

		if (node.loaded || node.loading || node.failed || node.isBackingOff())
		{
			return;
		}

		node.loading = true;
		// TODO: Need to put the numNodesLoading to the pco
		node.octreeGeometry.numNodesLoading++;

		try 
		{
			if (node.nodeType === 2)
			{ // TODO: Investigate
				await this.loadHierarchy(node);
			}

			let {byteOffset, byteSize} = node;
			
			if (byteOffset === undefined || byteSize === undefined) 
			{
				throw new Error('byteOffset and byteSize are required');
			}

			let urlOctree = (await this.requestManager.getUrl(this.url)).replace('/metadata.json', '/octree.bin');

			let first = byteOffset;
			let last = byteOffset + byteSize - BigInt(1);

			let buffer;

			if (byteSize === BigInt(0))
			{
				buffer = new ArrayBuffer(0);
				console.warn(`loaded node with 0 bytes: ${node.name}`);
			}
			else 
			{
				let response = await this.requestManager.fetch(urlOctree, {
					headers: {
						'content-type': 'multipart/byteranges',
						'Range': `bytes=${first}-${last}`
					}
				});

				buffer = await response.arrayBuffer();
			}

			const workerType = this.metadata.encoding === 'BROTLI' ? WorkerType.DECODER_WORKER_BROTLI : WorkerType.DECODER_WORKER;
			const worker = this.workerPool.getWorker(workerType);
			const detach = () =>
			{
				worker.onmessage = null;
				worker.onerror = null;
				worker.onmessageerror = null;
			};

			// A decoder that throws never posts a reply: without these handlers the
			// node stays "loading" and holds a slot of maxNumNodesLoading forever.
			worker.onerror = (event) =>
			{
				event.preventDefault();
				detach();
				// Its state after an uncaught exception is unknown: start a fresh one.
				worker.terminate();
				this.loadFailed(node, `decoder error: ${event.message}`);
			};
			worker.onmessageerror = () =>
			{
				detach();
				this.workerPool.returnWorker(workerType, worker);
				this.loadFailed(node, 'decoder reply could not be deserialized');
			};
			worker.onmessage = (e) =>
			{
				detach();
				this.workerPool.returnWorker(workerType, worker);
				try
				{
					this.onDecoded(node, e.data);
				}
				catch (error)
				{
					this.loadFailed(node, String(error));
				}
			};

			let pointAttributes = node.octreeGeometry.pointAttributes;
			let scale = node.octreeGeometry.scale;

			let box = node.boundingBox;
			let min = node.octreeGeometry.offset.clone().add(box.min);
			let size = box.max.clone().sub(box.min);
			let max = min.clone().add(size);
			let numPoints = node.numPoints;

			let offset = node.octreeGeometry.loader.offset;

			const injectFault = NodeLoader.debugFailNextDecodes > 0;
			if (injectFault)
			{
				NodeLoader.debugFailNextDecodes--;
			}

			let message = {
				name: node.name,
				buffer: buffer,
				pointAttributes: pointAttributes,
				scale: scale,
				min: min,
				max: max,
				size: size,
				offset: offset,
				numPoints: numPoints,
				debugThrow: injectFault
			};

			try
			{
				worker.postMessage(message, [message.buffer]);
			}
			catch (error)
			{
				detach();
				this.workerPool.returnWorker(workerType, worker);
				throw error;
			}
		}
		catch (e)
		{
			this.loadFailed(node, String(e));
		}
	}

	/**
	 * Turns a decoder reply into the node's geometry.
	 */
	private onDecoded(node: OctreeGeometryNode, data: any): void
	{
		if (node.octreeGeometry.disposed)
		{
			// The cloud was closed while this node decoded: keep nothing.
			node.loading = false;
			node.octreeGeometry.numNodesLoading = Math.max(0, node.octreeGeometry.numNodesLoading - 1);
			return;
		}

		let buffers = data.attributeBuffers;
		let geometry = new BufferGeometry();

		for (let property in buffers)
		{
			let buffer = buffers[property].buffer;
			let attribute: BufferAttribute;

			if (property === 'position')
			{
				attribute = new BufferAttribute(new Float32Array(buffer), 3);
			}
			else if (property === 'rgba')
			{
				attribute = new BufferAttribute(new Uint8Array(buffer), 4, true);
			}
			else if (property === 'NORMAL')
			{
				property = 'normal';
				attribute = new BufferAttribute(new Float32Array(buffer), 3);
			}
			else if (property === 'INDICES')
			{
				property = 'indices';
				attribute = new BufferAttribute(new Uint8Array(buffer), 4);
				attribute.normalized = true;
			}
			else
			{
				const scalar: BufferAttribute & {
					potree?: object
				} = new BufferAttribute(new Float32Array(buffer), 1);

				let batchAttribute = buffers[property].attribute;
				scalar.potree = {
					offset: buffers[property].offset,
					scale: buffers[property].scale,
					range: batchAttribute.range
				};
				attribute = scalar;
			}

			// Nothing on the CPU reads these once they are on the GPU (picking
			// reads positions, overlays rewrite classification in place).
			if (!KEEP_CPU_ATTRIBUTES.has(property))
			{
				releaseArrayOnUpload(attribute);
			}
			geometry.setAttribute(property, attribute);
		}
		node.density = data.density;
		node.geometry = geometry;
		node.loaded = true;
		node.loading = false;
		node.loadAttempts = 0;
		node.octreeGeometry.numNodesLoading = Math.max(0, node.octreeGeometry.numNodesLoading - 1);
		NodeLoader.stats.loaded++;

		// Invoke overlay hook so consumers can inject processed attributes. A
		// throwing hook must not turn a loaded node back into a failed one.
		if (NodeLoader.onNodeLoaded)
		{
			try
			{
				NodeLoader.onNodeLoaded(node, geometry);
			}
			catch (error)
			{
				console.error(`potree-core: onNodeLoaded threw for ${node.name}`, error);
			}
		}
	}

	/**
	 * Frees the node's loading slot so streaming goes on; after
	 * MAX_NODE_LOAD_ATTEMPTS failures the node is marked failed and skipped.
	 */
	private loadFailed(node: OctreeGeometryNode, reason: string): void
	{
		node.loaded = false;
		node.loading = false;
		node.octreeGeometry.numNodesLoading = Math.max(0, node.octreeGeometry.numNodesLoading - 1);
		node.loadAttempts++;
		// Back off before retrying, so a transient fault (I/O hiccup, worker
		// restart) does not burn every attempt within three frames.
		node.retryAfter = performance.now() + loadRetryDelayMs(node.loadAttempts);
		NodeLoader.stats.failures++;
		if (node.loadAttempts >= MAX_NODE_LOAD_ATTEMPTS)
		{
			node.failed = true;
		}
		console.warn(`potree-core: loading node ${node.name} failed (attempt ${node.loadAttempts}): ${reason}`);
	}

	public parseHierarchy(node: OctreeGeometryNode, buffer: ArrayBuffer)
	{
		let view = new DataView(buffer);
		let bytesPerNode = 22;
		let numNodes = buffer.byteLength / bytesPerNode;
		let octree = node.octreeGeometry;

		// let nodes = [node];
		let nodes: OctreeGeometryNode[] = new Array(numNodes);
		nodes[0] = node;
		let nodePos = 1;

		for (let i = 0; i < numNodes; i++)
		{
			let current = nodes[i];

			let type = view.getUint8(i * bytesPerNode + 0);
			let childMask = view.getUint8(i * bytesPerNode + 1);
			let numPoints = view.getUint32(i * bytesPerNode + 2, true);
			let byteOffset = view.getBigInt64(i * bytesPerNode + 6, true);
			let byteSize = view.getBigInt64(i * bytesPerNode + 14, true);


			if (current.nodeType === 2)
			{
				// replace proxy with real node
				current.byteOffset = byteOffset;
				current.byteSize = byteSize;
				current.numPoints = numPoints;
			}
			else if (type === 2)
			{
				// load proxy
				current.hierarchyByteOffset = byteOffset;
				current.hierarchyByteSize = byteSize;
				current.numPoints = numPoints;
			}
			else 
			{
				// load real node 
				current.byteOffset = byteOffset;
				current.byteSize = byteSize;
				current.numPoints = numPoints;
			}
			
			current.nodeType = type;

			if (current.nodeType === 2)
			{
				continue;
			}

			for (let childIndex = 0; childIndex < 8; childIndex++)
			{
				let childExists = (1 << childIndex & childMask) !== 0;

				if (!childExists)
				{
					continue;
				}

				let childName = current.name + childIndex;

				let childAABB = createChildAABB(current.boundingBox, childIndex);
				let child = new OctreeGeometryNode(childName, octree, childAABB);
				child.name = childName;
				child.spacing = current.spacing / 2;
				child.level = current.level + 1;

				(current.children as any)[childIndex] = child;
				child.parent = current;

				// nodes.push(child);
				nodes[nodePos] = child;
				nodePos++;
			}

			// if((i % 500) === 0){
			// 	yield;
			// }
		}
	}

	async loadHierarchy(node: OctreeGeometryNode)
	{

		let {hierarchyByteOffset, hierarchyByteSize} = node;

		if (hierarchyByteOffset === undefined || hierarchyByteSize === undefined) 
		{
			throw new Error(`hierarchyByteOffset and hierarchyByteSize are undefined for node ${node.name}`);
		}

		let hierarchyPath = (await this.requestManager.getUrl(this.url)).replace('/metadata.json', '/hierarchy.bin');

		let first = hierarchyByteOffset;
		let last = first + hierarchyByteSize - BigInt(1);

		let response = await this.requestManager.fetch(hierarchyPath, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${first}-${last}`
			}
		});

		let buffer = await response.arrayBuffer();

		this.parseHierarchy(node, buffer);
	}

}

let tmpVec3 = new Vector3();

/**
 * Creates a child AABB from the given parent AABB based on the specified index.
 * 
 * @param aabb - The parent AABB from which to create the child AABB.
 * @param index - The index of the child AABB to create, which determines its position relative to the parent AABB.
 * @returns The newly created child AABB.
 */
function createChildAABB(aabb: Box3, index: number)
{
	let min = aabb.min.clone();
	let max = aabb.max.clone();
	let size = tmpVec3.subVectors(max, min);

	if ((index & 0b0001) > 0) 
	{
		min.z += size.z / 2;
	}
	else 
	{
		max.z -= size.z / 2;
	}

	if ((index & 0b0010) > 0) 
	{
		min.y += size.y / 2;
	}
	else 
	{
		max.y -= size.y / 2;
	}
	
	if ((index & 0b0100) > 0) 
	{
		min.x += size.x / 2;
	}
	else 
	{
		max.x -= size.x / 2;
	}

	return new Box3(min, max);
}

let typenameTypeattributeMap = {
	'double': PointAttributeTypes.DATA_TYPE_DOUBLE,
	'float': PointAttributeTypes.DATA_TYPE_FLOAT,
	'int8': PointAttributeTypes.DATA_TYPE_INT8,
	'uint8': PointAttributeTypes.DATA_TYPE_UINT8,
	'int16': PointAttributeTypes.DATA_TYPE_INT16,
	'uint16': PointAttributeTypes.DATA_TYPE_UINT16,
	'int32': PointAttributeTypes.DATA_TYPE_INT32,
	'uint32': PointAttributeTypes.DATA_TYPE_UINT32,
	'int64': PointAttributeTypes.DATA_TYPE_INT64,
	'uint64': PointAttributeTypes.DATA_TYPE_UINT64
};

type AttributeType = keyof typeof typenameTypeattributeMap;

/**
 * Attribute interface defines the structure of an attribute in the octree geometry.
 */
export interface Attribute {
	name: string;
	description: string;
	size: number;
	numElements: number;
	type: AttributeType;
	min: number[];
	max: number[];
}

/**
 * Metadata interface defines the structure of the metadata for an octree geometry.
 */
export interface Metadata {
	version: string;
	name: string;
	description: string;
	points: number;
	projection: string;
	hierarchy: {
		firstChunkSize: number;
		stepSize: number;
		depth: number;
	},
	offset: [number, number, number],
	scale: [number, number, number],
	spacing: number,
	boundingBox: {
		min: [number, number, number],
		max: [number, number, number],
	},
	encoding: string;
	attributes: Attribute[];
}

/**
 * OctreeLoader is responsible for loading octree geometries from a given URL.
 */
export class OctreeLoader
{
	/**
	 * WorkerPool instance used for managing workers for loading tasks.
	 */
	public workerPool: WorkerPool = new WorkerPool();


	/**
	 * Parses the attributes from a JSON array and converts them into PointAttributes.
	 * 
	 * @param jsonAttributes Array of attributes in JSON format.
	 * @returns A PointAttributes instance containing the parsed attributes.
	 */
	public static parseAttributes(jsonAttributes: Attribute[]): PointAttributes
	{

		let attributes = new PointAttributes();

		// Replacements object for string to string
		let replacements: {[key: string]: string} = {'rgb': 'rgba'};

		for (const jsonAttribute of jsonAttributes) 
		{
			let {name, numElements, min, max} = jsonAttribute;

			let type = typenameTypeattributeMap[jsonAttribute.type]; // Fix the typing, currently jsonAttribute has type 'never'

			let potreeAttributeName = replacements[name] ? replacements[name] : name;

			let attribute = new PointAttribute(potreeAttributeName, type, numElements);

			if (numElements === 1)
			{
				attribute.range = [min[0], max[0]];
			}
			else 
			{
				attribute.range = [min, max];
			}

			if (name === 'gps-time') 
			{
				// HACK: Guard against bad gpsTime range in metadata, see potree/potree#909
				if (typeof attribute.range[0] === 'number' && attribute.range[0] === attribute.range[1]) 
				{
					attribute.range[1] += 1;
				}
			}

			attribute.initialRange = attribute.range;

			attributes.add(attribute);
		}

		{
			// check if it has normals
			let hasNormals = 
				attributes.attributes.find((a) => {return a.name === 'NormalX';}) !== undefined &&
				attributes.attributes.find((a) => {return a.name === 'NormalY';}) !== undefined &&
				attributes.attributes.find((a) => {return a.name === 'NormalZ';}) !== undefined;

			if (hasNormals)
			{
				let vector = {
					name: 'NORMAL',
					attributes: ['NormalX', 'NormalY', 'NormalZ']
				};
				attributes.addVector(vector);
			}
		}

		return attributes;
	}

	/**
	 * Loads an octree geometry from a given URL using the provided RequestManager.
	 * 
	 * @param url - The URL from which to load the octree geometry metadata. 
	 * @param requestManager - The RequestManager instance used to handle HTTP requests.
	 * @returns Geometry object containing the loaded octree geometry.
	 */
	public async load(url: string, requestManager: RequestManager)
	{

		let response = await requestManager.fetch(await requestManager.getUrl(url));
		let metadata: Metadata = await response.json();

		let attributes = OctreeLoader.parseAttributes(metadata.attributes);

		let loader = new NodeLoader(url, this.workerPool, metadata, requestManager);
		loader.attributes = attributes;
		loader.scale = metadata.scale;
		loader.offset = metadata.offset;

		let octree = new OctreeGeometry(loader, new Box3(new Vector3(...metadata.boundingBox.min), new Vector3(...metadata.boundingBox.max)));
		octree.url = await requestManager.getUrl(url);
		octree.spacing = metadata.spacing;
		octree.scale = metadata.scale;

		let min = new Vector3(...metadata.boundingBox.min);
		let max = new Vector3(...metadata.boundingBox.max);
		let boundingBox = new Box3(min, max);

		let offset = min.clone();
		boundingBox.min.sub(offset);
		boundingBox.max.sub(offset);

		octree.projection = metadata.projection;
		octree.boundingBox = boundingBox;
		// metadata.boundingBox is the cubic octree root; the real extent of the
		// data lives in the position attribute's min/max (PotreeConverter 2.x).
		const positionAttribute = (metadata.attributes as Array<{ name: string; min?: number[]; max?: number[] }>)
			.find((attribute) => attribute.name === 'position');
		const tightBoundingBox = positionAttribute?.min && positionAttribute?.max
			? new Box3(new Vector3(...positionAttribute.min).sub(offset), new Vector3(...positionAttribute.max).sub(offset))
			: boundingBox.clone();
		octree.tightBoundingBox = tightBoundingBox;
		octree.boundingSphere = boundingBox.getBoundingSphere(new Sphere());
		octree.tightBoundingSphere = tightBoundingBox.getBoundingSphere(new Sphere());
		octree.offset = offset;
		octree.pointAttributes = OctreeLoader.parseAttributes(metadata.attributes);

		let root = new OctreeGeometryNode('r', octree, boundingBox);
		root.level = 0;
		root.nodeType = 2;
		root.hierarchyByteOffset = BigInt(0);
		root.hierarchyByteSize = BigInt(metadata.hierarchy.firstChunkSize);
		root.spacing = octree.spacing;
		root.byteOffset = BigInt(0); // Originally 0

		octree.root = root;

		loader.load(root);

		let result = {geometry: octree};

		return result;

	}

}
