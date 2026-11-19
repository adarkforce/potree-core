import {
	AdditiveBlending,
	BufferGeometry,
	Camera,
	Color,
	GLSL3,
	LessEqualDepth,
	Material,
	NearestFilter,
	NoBlending,
	OrthographicCamera,
	PerspectiveCamera,
	RawShaderMaterial,
	Scene,
	Texture,
	Vector2,
	Vector3,
	Vector4,
	WebGLRenderer,
	WebGLRenderTarget
} from 'three';

import {
	DEFAULT_HIGHLIGHT_COLOR,
	DEFAULT_MAX_POINT_SIZE,
	DEFAULT_MIN_POINT_SIZE,
	DEFAULT_RGB_BRIGHTNESS,
	DEFAULT_RGB_CONTRAST,
	DEFAULT_RGB_GAMMA,
	PERSPECTIVE_CAMERA
} from '../constants';
import {PointCloudOctree} from '../point-cloud-octree';
import {PointCloudOctreeNode} from '../point-cloud-octree-node';
import {byLevelAndIndex} from '../utils/utils';
import {DEFAULT_CLASSIFICATION} from './classification';
import {ClipMode, IClipBox, IClipSphere} from './clipping';
import {PointColorType, PointOpacityType, PointShape, PointSizeType, TreeType} from './enums';
import {SPECTRAL} from './gradients';
import {
	generateClassificationTexture,
	generateDataTexture,
	generateGradientTexture
} from './texture-generation';
import {IClassification, IGradient, IUniform} from './types';
import {ColorEncoding} from './color-encoding';
import {packVisibleNodes, visibilityTextureRows, VISIBLE_NODES_TEXTURE_WIDTH} from './visibility-texture';
import {
	ClipAabbMode,
	FilterAttribute,
	MAX_ATTRIBUTE_FILTERS,
	NodeResultEntry,
	NodeResultState,
	ScalarSource,
	scalarFilterDomain,
	setClassBit,
	slopeToAbsNz
} from './style';

// @ts-ignore — Vite raw import
import VertShader from './shaders/pointcloud.vs?raw';
// @ts-ignore — Vite raw import
import FragShader from './shaders/pointcloud.fs?raw';

/**
 * Configuration parameters for point cloud material rendering.
 * 
 * @interface IPointCloudMaterialParameters
 */
export interface IPointCloudMaterialParameters {
	/**
	 * The base size of points in the point cloud.
	 */
	size: number;
	
	/**
	 * The minimum allowed size for points when scaling.
	 */
	minSize: number;
	
	/**
	 * The maximum allowed size for points when scaling.
	 */
	maxSize: number;
	
	/**
	 * The type of tree structure used for organizing the point cloud data.
	 */
	treeType: TreeType;
	
	/**
	 * Whether to use the new format for point cloud data processing.
	 */
	newFormat: boolean;
}


/**
 * Interface defining uniforms for point cloud material rendering in WebGL shaders.
 * These uniforms control various aspects of point cloud visualization including
 * appearance, filtering, transformations, and rendering parameters.
 * 
 * @interface IPointCloudMaterialUniforms
 */
export interface IPointCloudMaterialUniforms {
	/** Bounding box size as [width, height, depth] */
	bbSize: IUniform<[number, number, number]>;
	/** Supplement value for depth blending calculations */
	blendDepthSupplement: IUniform<number>;
	/** Hardness factor for blending operations */
	blendHardness: IUniform<number>;
	/** Lookup texture for point classification rendering */
	classificationLUT: IUniform<Texture>;
	/** Number of active clipping boxes */
	clipBoxCount: IUniform<number>;
	/** Array containing clipping box parameters */
	clipBoxes: IUniform<Float32Array>;
	/** Number of active clipping spheres */
	clipSphereCount: IUniform<number>;
	/** Array containing clipping sphere parameters (vec4: xyz=center, w=radius) */
	clipSpheres: IUniform<Float32Array>;
	/** Depth map texture for depth-based effects, null if not used */
	depthMap: IUniform<Texture | null>;
	/** Diffuse color as RGB values [r, g, b] */
	diffuse: IUniform<[number, number, number]>;
	/** Field of view angle in radians */
	fov: IUniform<number>;
	/** Gradient texture for color mapping */
	gradient: IUniform<Texture>;
	/** Maximum height value for elevation-based coloring */
	heightMax: IUniform<number>;
	/** Minimum height value for elevation-based coloring */
	heightMin: IUniform<number>;
	/** Brightness adjustment for intensity values */
	intensityBrightness: IUniform<number>;
	/** Contrast adjustment for intensity values */
	intensityContrast: IUniform<number>;
	/** Gamma correction for intensity values */
	intensityGamma: IUniform<number>;
	/** Intensity range as [min, max] values */
	intensityRange: IUniform<[number, number]>;
	/** Current level of detail */
	level: IUniform<number>;
	/** Maximum point size in pixels */
	maxSize: IUniform<number>;
	/** Minimum point size in pixels */
	minSize: IUniform<number>;
	/** Size of the octree structure */
	octreeSize: IUniform<number>;
	/** Overall opacity of the point cloud (0.0 to 1.0) */
	opacity: IUniform<number>;
	/** Point cloud index identifier */
	pcIndex: IUniform<number>;
	/** Brightness adjustment for RGB color values */
	rgbBrightness: IUniform<number>;
	/** Contrast adjustment for RGB color values */
	rgbContrast: IUniform<number>;
	/** Gamma correction for RGB color values */
	rgbGamma: IUniform<number>;
	/** Screen height in pixels */
	screenHeight: IUniform<number>;
	/** Screen width in pixels */
	screenWidth: IUniform<number>;
	/** Orthographic camera height */
	orthoHeight: IUniform<number>;
	/** Orthographic camera width */
	orthoWidth: IUniform<number>;
	/** Flag indicating whether orthographic camera is being used */
	useOrthographicCamera: IUniform<boolean>;
	/** Far clipping plane distance */
	far: IUniform<number>;
	/** Base point size */
	size: IUniform<number>;
	/** Spacing between points */
	spacing: IUniform<number>;
	/** Transformation matrix to model space */
	toModel: IUniform<number[]>;
	/** Transition factor for animations or interpolations */
	transition: IUniform<number>;
	/** Color uniform for material */
	uColor: IUniform<Color>;
	/** Texture containing visible node information */
	visibleNodes: IUniform<Texture>;
	/** Starting index for visible nodes */
	vnStart: IUniform<number>;
	/** Weight factor for classification-based coloring */
	wClassification: IUniform<number>;
	/** Weight factor for elevation-based coloring */
	wElevation: IUniform<number>;
	/** Weight factor for intensity-based coloring */
	wIntensity: IUniform<number>;
	/** Weight factor for return number-based coloring */
	wReturnNumber: IUniform<number>;
	/** Weight factor for RGB color contribution */
	wRGB: IUniform<number>;
	/** Weight factor for source ID-based coloring */
	wSourceID: IUniform<number>;
	/** Opacity attenuation factor based on distance or other criteria */
	opacityAttenuation: IUniform<number>;
	/** Threshold value for normal-based point filtering */
	filterByNormalThreshold: IUniform<number>;
	/** 3D coordinate of the highlighted point */
	highlightedPointCoordinate: IUniform<Vector3>;
	/** RGBA color for highlighted point rendering */
	highlightedPointColor: IUniform<Vector4>;
	/** Flag to enable or disable point highlighting feature */
	enablePointHighlighting: IUniform<boolean>;
	/** Scale factor for highlighted point size */
	highlightedPointScale: IUniform<number>;
	/** Scale factor for view-dependent sizing */
	viewScale: IUniform<number>;
	/** Minimum value for overlay scalar range mapping */
	overlayMin: IUniform<number>;
	/** Maximum value for overlay scalar range mapping */
	overlayMax: IUniform<number>;
	/** Points whose overlayScalar lies outside [x, y] are culled while scalarFilterEnabled is 1. */
	scalarFilterRange: IUniform<Vector2>;
	/** 1 while the scalar range filter is on. */
	scalarFilterEnabled: IUniform<number>;
	/** Per node: 1 when its geometry carries overlayScalar (the filter leaves the others alone). */
	hasOverlayScalar: IUniform<number>;
	/** Live style (see style.ts): 256-bit hidden-class mask, every colour mode. */
	classHidden: IUniform<Int32Array>;
	/** Attribute filter slots: FilterAttribute per slot (0 = off). */
	attrFilterAttr: IUniform<Int32Array>;
	/** Attribute filter slots: [low, high] pairs, stored units. */
	attrFilterRange: IUniform<Float32Array>;
	/** ClipAabbMode: 0 off, 1 keep inside, 2 keep outside. */
	clipAabbMode: IUniform<number>;
	/** Clip box min, cloud frame. */
	clipAabbMin: IUniform<Vector3>;
	/** Clip box max, cloud frame. */
	clipAabbMax: IUniform<Vector3>;
	/** Per node: its box min in the cloud frame. */
	nodeOffset: IUniform<Vector3>;
	/** ScalarSource. */
	scalarSource: IUniform<number>;
	/** CovarianceFeature shown when scalarSource is FEATURE. */
	featureIndex: IUniform<number>;
	/** scalarFilterRange mapped to the exact test's domain (scalarFilterDomain()). */
	scalarFilterDomain: IUniform<Vector2>;
	/** 1 while the slope filter over stored normals is on. */
	normalFilterEnabled: IUniform<number>;
	/** Slope filter as a |n.z| range. */
	normalFilterAbsNz: IUniform<Vector2>;
	/** Per node: 1 when the geometry carries overlayEigen. */
	hasOverlayEigen: IUniform<number>;
	/** Per node: 1 when the geometry carries normal. */
	hasNormal: IUniform<number>;
	/** Per node: 1 when the geometry carries rgba. */
	hasRgba: IUniform<number>;
	/** 1: apply rgbGamma / rgbBrightness / rgbContrast (no recompile). */
	rgbAdjust: IUniform<number>;
	/** Per node: NodeResultState. */
	nodeResultState: IUniform<number>;
	/** Per node: ms since its result arrived. */
	resultAge: IUniform<number>;
	/** 1: mark preview nodes (desaturated, shimmer). */
	previewMarks: IUniform<number>;
	/** 1: no shimmer (prefers-reduced-motion). */
	reducedMotion: IUniform<number>;
	/** Seconds; drives the preview shimmer. */
	styleTime: IUniform<number>;
}

/** A closed range of overlayScalar values left visible. */
export type ScalarFilter = Readonly<{low: number; high: number}>;

const TREE_TYPE_DEFS = {
	[TreeType.OCTREE]: 'tree_type_octree',
	[TreeType.KDTREE]: 'tree_type_kdtree'
};

const SIZE_TYPE_DEFS = {
	[PointSizeType.FIXED]: 'fixed_point_size',
	[PointSizeType.ATTENUATED]: 'attenuated_point_size',
	[PointSizeType.ADAPTIVE]: 'adaptive_point_size'
};

const OPACITY_DEFS = {
	[PointOpacityType.ATTENUATED]: 'attenuated_opacity',
	[PointOpacityType.FIXED]: 'fixed_opacity'
};

const SHAPE_DEFS = {
	[PointShape.SQUARE]: 'square_point_shape',
	[PointShape.CIRCLE]: 'circle_point_shape',
	[PointShape.PARABOLOID]: 'paraboloid_point_shape'
};

const COLOR_DEFS = {
	[PointColorType.RGB]: 'color_type_rgb',
	[PointColorType.COLOR]: 'color_type_color',
	[PointColorType.DEPTH]: 'color_type_depth',
	[PointColorType.HEIGHT]: 'color_type_height',
	[PointColorType.INTENSITY]: 'color_type_intensity',
	[PointColorType.INTENSITY_GRADIENT]: 'color_type_intensity_gradient',
	[PointColorType.LOD]: 'color_type_lod',
	[PointColorType.POINT_INDEX]: 'color_type_point_index',
	[PointColorType.CLASSIFICATION]: 'color_type_classification',
	[PointColorType.RETURN_NUMBER]: 'color_type_return_number',
	[PointColorType.SOURCE]: 'color_type_source',
	[PointColorType.NORMAL]: 'color_type_normal',
	[PointColorType.PHONG]: 'color_type_phong',
	[PointColorType.RGB_HEIGHT]: 'color_type_rgb_height',
	[PointColorType.OVERLAY_SCALAR]: 'color_type_overlay_scalar',
	[PointColorType.COMPOSITE]: 'color_type_composite'
};

const CLIP_MODE_DEFS = {
	[ClipMode.DISABLED]: 'clip_disabled',
	[ClipMode.CLIP_OUTSIDE]: 'clip_outside',
	[ClipMode.CLIP_INSIDE]: 'clip_inside',
	[ClipMode.HIGHLIGHT_INSIDE]: 'clip_highlight_inside'
};

const INPUT_COLOR_ENCODING = {
	[ColorEncoding.LINEAR]: 'input_color_encoding_linear',
	[ColorEncoding.SRGB]: 'input_color_encoding_sRGB'
};

const OUTPUT_COLOR_ENCODING = {
	[ColorEncoding.LINEAR]: 'output_color_encoding_linear',
	[ColorEncoding.SRGB]: 'output_color_encoding_sRGB'
};

export class PointCloudMaterial extends RawShaderMaterial 
{
	private static helperVec3 = new Vector3();

	private static helperVec2 = new Vector2();

	/** Texel data of the visible-nodes texture, reused across frames. */
	private visibleNodesData: Uint8Array = new Uint8Array(0);

	lights = false;

	fog = false;

	numClipBoxes: number = 0;

	clipBoxes: IClipBox[] = [];

	numClipSpheres: number = 0;

	clipSpheres: IClipSphere[] = [];

	visibleNodesTexture: Texture | undefined;

	private visibleNodeTextureOffsets = new Map<string, number>();

	private _gradient = SPECTRAL;

	private gradientTexture: Texture | undefined = generateGradientTexture(this._gradient);

	private _classification: IClassification = DEFAULT_CLASSIFICATION;

	private classificationTexture: Texture | undefined = generateClassificationTexture(
		this._classification,
	);

	uniforms: IPointCloudMaterialUniforms & Record<string, IUniform<any>> = {
		bbSize: makeUniform('fv', [0, 0, 0] as [number, number, number]),
		blendDepthSupplement: makeUniform('f', 0.0),
		blendHardness: makeUniform('f', 2.0),
		classificationLUT: makeUniform('t', this.classificationTexture || new Texture()),
		clipBoxCount: makeUniform('f', 0),
		clipBoxes: makeUniform('Matrix4fv', [] as any),
		clipSphereCount: makeUniform('f', 0),
		clipSpheres: makeUniform('fv', [] as any),
		depthMap: makeUniform('t', null),
		diffuse: makeUniform('fv', [1, 1, 1] as [number, number, number]),
		fov: makeUniform('f', 1.0),
		gradient: makeUniform('t', this.gradientTexture || new Texture()),
		heightMax: makeUniform('f', 1.0),
		heightMin: makeUniform('f', 0.0),
		intensityBrightness: makeUniform('f', 0),
		intensityContrast: makeUniform('f', 0),
		intensityGamma: makeUniform('f', 1),
		intensityRange: makeUniform('fv', [0, 65000] as [number, number]),
		isLeafNode: makeUniform('b', 0),
		level: makeUniform('f', 0.0),
		maxSize: makeUniform('f', DEFAULT_MAX_POINT_SIZE),
		minSize: makeUniform('f', DEFAULT_MIN_POINT_SIZE),
		octreeSize: makeUniform('f', 0),
		opacity: makeUniform('f', 1.0),
		pcIndex: makeUniform('f', 0),
		rgbBrightness: makeUniform('f', DEFAULT_RGB_BRIGHTNESS),
		rgbContrast: makeUniform('f', DEFAULT_RGB_CONTRAST),
		rgbGamma: makeUniform('f', DEFAULT_RGB_GAMMA),
		screenHeight: makeUniform('f', 1.0),
		screenWidth: makeUniform('f', 1.0),
		useOrthographicCamera: makeUniform('b', false),
		orthoHeight: makeUniform('f', 1.0),
		orthoWidth: makeUniform('f', 1.0),
		far: makeUniform('f', 1000.0),
		size: makeUniform('f', 1),
		spacing: makeUniform('f', 1.0),
		toModel: makeUniform('Matrix4f', []),
		transition: makeUniform('f', 0.5),
		uColor: makeUniform('c', new Color(0xffffff)),
		// @ts-ignore
		visibleNodes: makeUniform('t', this.visibleNodesTexture || new Texture()),
		vnStart: makeUniform('f', 0.0),
		wClassification: makeUniform('f', 0),
		wElevation: makeUniform('f', 0),
		wIntensity: makeUniform('f', 0),
		wReturnNumber: makeUniform('f', 0),
		wRGB: makeUniform('f', 1),
		wSourceID: makeUniform('f', 0),
		opacityAttenuation: makeUniform('f', 1),
		filterByNormalThreshold: makeUniform('f', 0),
		highlightedPointCoordinate: makeUniform('fv', new Vector3()),
		highlightedPointColor: makeUniform('fv', DEFAULT_HIGHLIGHT_COLOR.clone()),
		enablePointHighlighting: makeUniform('b', true),
		highlightedPointScale: makeUniform('f', 2.0),
		viewScale: makeUniform('f', 1.0),
		overlayMin: makeUniform('f', 0.0),
		overlayMax: makeUniform('f', 1.0),
		scalarFilterRange: makeUniform('fv', new Vector2(-Infinity, Infinity)),
		scalarFilterEnabled: makeUniform('f', 0),
		hasOverlayScalar: makeUniform('f', 0),
		classHidden: makeUniform('iv1', new Int32Array(8)),
		attrFilterAttr: makeUniform('iv1', new Int32Array(MAX_ATTRIBUTE_FILTERS)),
		attrFilterRange: makeUniform('fv', new Float32Array(2 * MAX_ATTRIBUTE_FILTERS)),
		clipAabbMode: makeUniform('f', 0),
		clipAabbMin: makeUniform('fv', new Vector3(-Infinity, -Infinity, -Infinity)),
		clipAabbMax: makeUniform('fv', new Vector3(Infinity, Infinity, Infinity)),
		nodeOffset: makeUniform('fv', new Vector3()),
		scalarSource: makeUniform('i', 0),
		featureIndex: makeUniform('i', 0),
		scalarFilterDomain: makeUniform('fv', new Vector2(-Infinity, Infinity)),
		normalFilterEnabled: makeUniform('f', 0),
		normalFilterAbsNz: makeUniform('fv', new Vector2(-Infinity, Infinity)),
		hasOverlayEigen: makeUniform('f', 0),
		hasNormal: makeUniform('f', 0),
		hasRgba: makeUniform('f', 0),
		rgbAdjust: makeUniform('f', 0),
		nodeResultState: makeUniform('f', 0),
		resultAge: makeUniform('f', 1e9),
		previewMarks: makeUniform('f', 1),
		reducedMotion: makeUniform('f', 0),
		styleTime: makeUniform('f', 0)
	};

	/** Live result state per node name (shared with copies made by copyStyleFrom). */
	nodeResultStates: Map<string, NodeResultEntry> = new Map();

	/** State of nodes absent from nodeResultStates. */
	defaultNodeResultState: NodeResultState = NodeResultState.UNTRACKED;

  @uniform('bbSize') bbSize!: [number, number, number];

  @uniform('depthMap') depthMap!: Texture | undefined;

  @uniform('fov') fov!: number;

  @uniform('heightMax') heightMax!: number;

  @uniform('heightMin') heightMin!: number;

  @uniform('intensityBrightness') intensityBrightness!: number;

  @uniform('intensityContrast') intensityContrast!: number;

  @uniform('intensityGamma') intensityGamma!: number;

  @uniform('intensityRange') intensityRange!: [number, number];

  @uniform('maxSize') maxSize!: number;

  @uniform('minSize') minSize!: number;

  @uniform('octreeSize') octreeSize!: number;

  @uniform('opacity', true) opacity!: number;

  // Uniform-only: the rgbAdjust switch replaces the old define, so a slider
  // drag never recompiles (set the rgbAdjust uniform to 1 to apply them).
  @uniform('rgbBrightness') rgbBrightness!: number;

  @uniform('rgbContrast') rgbContrast!: number;

  @uniform('rgbGamma') rgbGamma!: number;

  @uniform('screenHeight') screenHeight!: number;

  @uniform('screenWidth') screenWidth!: number;

  @uniform('orthoWidth') orthoWidth!: number;

  @uniform('orthoHeight') orthoHeight!: number;

  @uniform('useOrthographicCamera') useOrthographicCamera!: boolean;
  
  @uniform('far') far!: number;

  @uniform('size') size!: number;

  @uniform('spacing') spacing!: number;

  @uniform('transition') transition!: number;

  @uniform('uColor') color!: Color;

  @uniform('wClassification') weightClassification!: number;

  @uniform('wElevation') weightElevation!: number;

  @uniform('wIntensity') weightIntensity!: number;

  @uniform('wReturnNumber') weightReturnNumber!: number;

  @uniform('wRGB') weightRGB!: number;

  @uniform('wSourceID') weightSourceID!: number;

  @uniform('opacityAttenuation') opacityAttenuation!: number;

  @uniform('filterByNormalThreshold') filterByNormalThreshold!: number;

  @uniform('highlightedPointCoordinate') highlightedPointCoordinate!: Vector3;

  @uniform('highlightedPointColor') highlightedPointColor!: Vector4;

  @uniform('enablePointHighlighting') enablePointHighlighting!: boolean;

  @uniform('highlightedPointScale') highlightedPointScale!: number;

  @uniform('viewScale') viewScale!: number;

  @uniform('overlayMin') overlayMin!: number;

  @uniform('overlayMax') overlayMax!: number;

  /**
   * The scalar range filter: points of nodes that carry overlayScalar and whose
   * value lies outside [low, high] are culled in the vertex shader; null shows
   * every point. Setting it updates two uniforms: no allocation, no upload.
   */
  get scalarFilter(): ScalarFilter | null
  {
  	if (this.uniforms.scalarFilterEnabled.value < 0.5)
  	{
  		return null;
  	}
  	const range = this.uniforms.scalarFilterRange.value;
  	return {low: range.x, high: range.y};
  }

  set scalarFilter(filter: ScalarFilter | null)
  {
  	if (filter)
  	{
  		this.uniforms.scalarFilterRange.value.set(filter.low, filter.high);
  		this.syncScalarFilterDomain();
  	}
  	this.uniforms.scalarFilterEnabled.value = filter ? 1 : 0;
  	this.uniformsNeedUpdate = true;
  }

  /** Turns the filter on with [low, high] without allocating a filter object (slider drags). */
  setScalarFilterRange(low: number, high: number): void
  {
  	this.uniforms.scalarFilterRange.value.set(low, high);
  	this.syncScalarFilterDomain();
  	this.uniforms.scalarFilterEnabled.value = 1;
  	this.uniformsNeedUpdate = true;
  }

  /** Recomputes scalarFilterDomain from the range, the scalar source and the feature. */
  private syncScalarFilterDomain(): void
  {
  	const range = this.uniforms.scalarFilterRange.value;
  	const [lo, hi] = scalarFilterDomain(this.uniforms.scalarSource.value, this.uniforms.featureIndex.value, range.x, range.y);
  	this.uniforms.scalarFilterDomain.value.set(lo, hi);
  }

  /**
   * What the scalar colour mode and the scalar filter read: overlayScalar, a
   * covariance feature of overlayEigen (Select Field), slope or verticality
   * of the stored normal. Uniform-only.
   */
  setScalarSource(source: ScalarSource, feature: number = 0): void
  {
  	this.uniforms.scalarSource.value = source;
  	this.uniforms.featureIndex.value = Math.trunc(feature);
  	this.syncScalarFilterDomain();
  	this.uniformsNeedUpdate = true;
  }

  /** Hides (or shows) points of the given class in every colour mode. */
  setClassHidden(cls: number, hidden: boolean): void
  {
  	setClassBit(this.uniforms.classHidden.value, cls, hidden);
  	this.uniformsNeedUpdate = true;
  }

  /** Replaces the hidden-class set. */
  setHiddenClasses(classes: Iterable<number>): void
  {
  	const mask = this.uniforms.classHidden.value;
  	mask.fill(0);
  	for (const c of classes)
  	{
  		setClassBit(mask, c, true);
  	}
  	this.uniformsNeedUpdate = true;
  }

  /** Attribute range filter in slot 0..3: keep [low, high] of `attribute` (stored units); OFF clears it. */
  setAttributeFilter(slot: number, attribute: FilterAttribute, low: number = -Infinity, high: number = Infinity): void
  {
  	if (slot < 0 || slot >= MAX_ATTRIBUTE_FILTERS)
  	{
  		throw new RangeError(`attribute filter slot ${slot} out of [0, ${MAX_ATTRIBUTE_FILTERS})`);
  	}
  	this.uniforms.attrFilterAttr.value[slot] = attribute;
  	this.uniforms.attrFilterRange.value[2 * slot] = low;
  	this.uniforms.attrFilterRange.value[2 * slot + 1] = high;
  	this.uniformsNeedUpdate = true;
  }

  /** Axis-aligned clip box in the cloud frame (PassThrough, height range); OFF disables it. */
  setClipAabb(mode: ClipAabbMode, min?: Vector3, max?: Vector3): void
  {
  	this.uniforms.clipAabbMode.value = mode;
  	if (min)
  	{
  		this.uniforms.clipAabbMin.value.copy(min);
  	}
  	if (max)
  	{
  		this.uniforms.clipAabbMax.value.copy(max);
  	}
  	this.uniformsNeedUpdate = true;
  }

  /** Slope filter over stored normals: keep slopes in [lowDeg, highDeg]; null disables it. */
  setNormalFilter(range: Readonly<{lowDeg: number; highDeg: number}> | null): void
  {
  	if (range)
  	{
  		const [lo, hi] = slopeToAbsNz(range.lowDeg, range.highDeg);
  		this.uniforms.normalFilterAbsNz.value.set(lo, hi);
  	}
  	this.uniforms.normalFilterEnabled.value = range ? 1 : 0;
  	this.uniformsNeedUpdate = true;
  }

  /** Per-node live result state; `since` is the performance.now() of its arrival (drives the fade-in). */
  setNodeResultState(nodeName: string, state: NodeResultState, since: number): void
  {
  	this.nodeResultStates.set(nodeName, {state, since});
  }

  /** Forgets every node's state; nodes then draw as `defaultState`. */
  resetNodeResultStates(defaultState: NodeResultState = NodeResultState.UNTRACKED): void
  {
  	this.nodeResultStates.clear();
  	this.defaultNodeResultState = defaultState;
  }

  /**
   * Copies the live style onto another material of the same cloud (the HQ
   * splat renderer's passes): uniform values, and the node states by
   * reference. Allocation-free; call it every pass.
   */
  copyStyleFrom(source: PointCloudMaterial): void
  {
  	const s = source.uniforms;
  	const t = this.uniforms;
  	t.classHidden.value.set(s.classHidden.value);
  	t.attrFilterAttr.value.set(s.attrFilterAttr.value);
  	t.attrFilterRange.value.set(s.attrFilterRange.value);
  	t.clipAabbMode.value = s.clipAabbMode.value;
  	t.clipAabbMin.value.copy(s.clipAabbMin.value);
  	t.clipAabbMax.value.copy(s.clipAabbMax.value);
  	t.scalarSource.value = s.scalarSource.value;
  	t.featureIndex.value = s.featureIndex.value;
  	t.scalarFilterRange.value.copy(s.scalarFilterRange.value);
  	t.scalarFilterDomain.value.copy(s.scalarFilterDomain.value);
  	t.scalarFilterEnabled.value = s.scalarFilterEnabled.value;
  	t.normalFilterEnabled.value = s.normalFilterEnabled.value;
  	t.normalFilterAbsNz.value.copy(s.normalFilterAbsNz.value);
  	t.rgbAdjust.value = s.rgbAdjust.value;
  	t.previewMarks.value = s.previewMarks.value;
  	t.reducedMotion.value = s.reducedMotion.value;
  	t.styleTime.value = s.styleTime.value;
  	this.nodeResultStates = source.nodeResultStates;
  	this.defaultNodeResultState = source.defaultNodeResultState;
  	this.uniformsNeedUpdate = true;
  }

  // Declare PointCloudMaterial attributes that need shader updates upon change, and set default values.
  @requiresShaderUpdate() useClipBox: boolean = false;

  @requiresShaderUpdate() useClipSphere: boolean = false;

  @requiresShaderUpdate() weighted: boolean = false;

  @requiresShaderUpdate() pointColorType: PointColorType = PointColorType.RGB;

  @requiresShaderUpdate() pointSizeType: PointSizeType = PointSizeType.ADAPTIVE;

  @requiresShaderUpdate() clipMode: ClipMode = ClipMode.DISABLED;

  @requiresShaderUpdate() useEDL: boolean = false;

  @requiresShaderUpdate() shape: PointShape = PointShape.SQUARE;

  @requiresShaderUpdate() treeType: TreeType = TreeType.OCTREE;

  @requiresShaderUpdate() pointOpacityType: PointOpacityType = PointOpacityType.FIXED;

  @requiresShaderUpdate() useFilterByNormal: boolean = false;

  @requiresShaderUpdate() highlightPoint: boolean = false;

  @requiresShaderUpdate() inputColorEncoding: ColorEncoding = ColorEncoding.SRGB;

  @requiresShaderUpdate() outputColorEncoding: ColorEncoding = ColorEncoding.LINEAR;

  @requiresShaderUpdate() private useLogDepth: boolean = false;

  attributes = {
  	position: {type: 'fv', value: []},
  	color: {type: 'fv', value: []},
  	normal: {type: 'fv', value: []},
  	intensity: {type: 'f', value: []},
  	classification: {type: 'f', value: []},
  	returnNumber: {type: 'f', value: []},
  	numberOfReturns: {type: 'f', value: []},
  	pointSourceID: {type: 'f', value: []},
  	indices: {type: 'fv', value: []}
  };

  newFormat: boolean;

  constructor(parameters: Partial<IPointCloudMaterialParameters> = {}) 
  {
  	super();

  	this.allocateVisibleNodesTexture(1);

  	this.treeType = getValid(parameters.treeType, TreeType.OCTREE);
  	this.size = getValid(parameters.size, 1.0);
  	this.minSize = getValid(parameters.minSize, 2.0);
  	this.maxSize = getValid(parameters.maxSize, 50.0);

  	this.newFormat = Boolean(parameters.newFormat);

  	this.classification = DEFAULT_CLASSIFICATION;

  	(this.defaultAttributeValues as Record<string, number[]>).normal = [0, 0, 0];
  	(this.defaultAttributeValues as Record<string, number[]>).classification = [0, 0, 0];
  	(this.defaultAttributeValues as Record<string, number[]>).overlayScalar = [0];
  	(this.defaultAttributeValues as Record<string, number[]>).overlayVisible = [1];
  	(this.defaultAttributeValues as Record<string, number[]>).overlayRGBA = [0, 0, 0, 0];
  	(this.defaultAttributeValues as Record<string, number[]>).indices = [0, 0, 0, 0];
  	// Deterministic values for attributes a cloud lacks (the CPU twin reads 0 too).
  	for (const name of ['intensity', 'returnNumber', 'numberOfReturns', 'pointSourceID', 'gpsTime'])
  	{
  		(this.defaultAttributeValues as Record<string, number[]>)[name] = [0];
  	}
  	(this.defaultAttributeValues as Record<string, number[]>).overlayEigen = [0, 0, 0, 0];

  	this.vertexColors = true;
	
  	// throw new Error('Not implemented');
  	// this.extensions.fragDepth = true;

  	this.updateShaderSource();
  }

  dispose(): void 
  {
  	super.dispose();

  	if (this.gradientTexture) 
  	{
  		this.gradientTexture.dispose();
  		this.gradientTexture = undefined;
  	}

  	if (this.visibleNodesTexture)
  	{
  		this.visibleNodesTexture.dispose();
  		this.visibleNodesTexture = undefined;
  		this.visibleNodesData = new Uint8Array(0);
  	}

  	this.clearVisibleNodeTextureOffsets();

  	if (this.classificationTexture) 
  	{
  		this.classificationTexture.dispose();
  		this.classificationTexture = undefined;
  	}

  	if (this.depthMap) 
  	{
  		this.depthMap.dispose();
  		this.depthMap = undefined;
  	}
  }

  clearVisibleNodeTextureOffsets(): void 
  {
  	this.visibleNodeTextureOffsets.clear();
  }

  updateShaderSource(): void 
  {
  	this.glslVersion = GLSL3;

  	this.vertexShader = this.applyDefines(VertShader);
  	this.fragmentShader = this.applyDefines(FragShader);

  	if (this.opacity === 1.0) 
  	{
  		this.blending = NoBlending;
  		this.transparent = false;
  		this.depthTest = true;
  		this.depthWrite = true;
  		this.depthFunc = LessEqualDepth;
  	}
  	else if (this.opacity < 1.0 && !this.useEDL) 
  	{
  		this.blending = AdditiveBlending;
  		this.transparent = true;
  		this.depthTest = false;
  		this.depthWrite = true;
  	}

  	if (this.weighted) 
  	{
  		this.blending = AdditiveBlending;
  		this.transparent = true;
  		this.depthTest = true;
  		this.depthWrite = false;
  		this.depthFunc = LessEqualDepth;
  	}

  	this.needsUpdate = true;
  }

  applyDefines(shaderSrc: string): string 
  {
  	const parts: string[] = [];

  	function define(value: string | undefined) 
  	{
  		if (value) 
  		{
  			parts.push(`#define ${value}`);
  		}
  	}

  	define(TREE_TYPE_DEFS[this.treeType]);
  	define(SIZE_TYPE_DEFS[this.pointSizeType]);
  	define(SHAPE_DEFS[this.shape]);
  	define(COLOR_DEFS[this.pointColorType]);
  	define(CLIP_MODE_DEFS[this.clipMode]);
  	define(OPACITY_DEFS[this.pointOpacityType]);
  	define(OUTPUT_COLOR_ENCODING[this.outputColorEncoding]);
  	define(INPUT_COLOR_ENCODING[this.inputColorEncoding]);

  	if (this.useFilterByNormal) 
  	{
  		define('use_filter_by_normal');
  	}

  	if (this.useEDL) 
  	{
  		define('use_edl');
  	}

	if (this.useLogDepth) 
	{
		define('use_log_depth');
	}

  	if (this.weighted) 
  	{
  		define('weighted_splats');
  	}

  	if (this.numClipBoxes > 0) 
  	{
  		define('use_clip_box');
  	}

  	if (this.numClipSpheres > 0)
  	{
  		define('use_clip_sphere');
  	}

  	if (this.highlightPoint) 
  	{
  		define('highlight_point');
  	}

  	define('MAX_POINT_LIGHTS 0');
  	define('MAX_DIR_LIGHTS 0');

  	if (this.newFormat) 
  	{
  		define ('new_format');
  	}


  	// If '#version 300 es' exists as a line in shaderSrc, remove it and add it as the first element in the parts array
  	const versionLine = shaderSrc.match(/^\s*#version\s+300\s+es\s*\n/);
  	if (versionLine) 
  	{
  		parts.unshift(versionLine[0]);
  		shaderSrc = shaderSrc.replace(versionLine[0], '');
  	}
  	parts.push(shaderSrc);
  	return parts.join('\n');
  }

  setClipBoxes(clipBoxes: IClipBox[]): void 
  {
  	if (!clipBoxes) 
  	{
  		return;
  	}

  	this.clipBoxes = clipBoxes;

  	const doUpdate =
	  this.numClipBoxes !== clipBoxes.length && (clipBoxes.length === 0 || this.numClipBoxes === 0);

  	this.numClipBoxes = clipBoxes.length;
  	this.setUniform('clipBoxCount', this.numClipBoxes);

  	if (doUpdate) 
  	{
  		this.updateShaderSource();
  	}

  	const clipBoxesLength = this.numClipBoxes * 16;
  	const clipBoxesArray = new Float32Array(clipBoxesLength);

  	for (let i = 0; i < this.numClipBoxes; i++) 
  	{
  		clipBoxesArray.set(clipBoxes[i].inverse.elements, 16 * i);
  	}

  	for (let i = 0; i < clipBoxesLength; i++) 
  	{
  		if (isNaN(clipBoxesArray[i])) 
  		{
  			clipBoxesArray[i] = Infinity;
  		}
  	}

  	this.setUniform('clipBoxes', clipBoxesArray);
  }

  setClipSpheres(clipSpheres: IClipSphere[]): void
  {
  	if (!clipSpheres)
  	{
  		return;
  	}

  	this.clipSpheres = clipSpheres;

  	const doUpdate =
	  (this.numClipSpheres === 0) !== (clipSpheres.length === 0);

  	this.numClipSpheres = clipSpheres.length;
  	this.setUniform('clipSphereCount', this.numClipSpheres);

  	if (doUpdate)
  	{
  		this.updateShaderSource();
  	}

  	const clipSpheresLength = this.numClipSpheres * 4;
  	const clipSpheresArray = new Float32Array(clipSpheresLength);

  	for (let i = 0; i < this.numClipSpheres; i++)
  	{
  		clipSpheresArray[i * 4 + 0] = clipSpheres[i].center.x;
  		clipSpheresArray[i * 4 + 1] = clipSpheres[i].center.y;
  		clipSpheresArray[i * 4 + 2] = clipSpheres[i].center.z;
  		clipSpheresArray[i * 4 + 3] = clipSpheres[i].radius;
  	}

  	this.setUniform('clipSpheres', clipSpheresArray);
  }

  get gradient(): IGradient 
  {
  	return this._gradient;
  }

  set gradient(value: IGradient) 
  {
  	if (this._gradient !== value) 
  	{
  		this._gradient = value;
  		this.gradientTexture = generateGradientTexture(this._gradient);
  		this.setUniform('gradient', this.gradientTexture);
  	}
  }

  get classification(): IClassification 
  {
  	return this._classification;
  }

  set classification(value: IClassification) 
  {
  	const copy: IClassification = {} as any;
  	for (const key of Object.keys(value)) 
  	{
  		copy[key] = value[key].clone();
  	}

  	let isEqual = false;
  	if (this._classification === undefined) 
  	{
  		isEqual = false;
  	}
  	else 
  	{
  		isEqual = Object.keys(copy).length === Object.keys(this._classification).length;

  		for (const key of Object.keys(copy)) 
  		{
  			isEqual = isEqual && this._classification[key] !== undefined;
  			isEqual = isEqual && copy[key].equals(this._classification[key]);
  		}
  	}

  	if (!isEqual) 
  	{
  		this._classification = copy;
  		this.recomputeClassification();
  	}
  }

  private recomputeClassification(): void 
  {
  	this.classificationTexture = generateClassificationTexture(this._classification);
  	this.setUniform('classificationLUT', this.classificationTexture);
  }

  get elevationRange(): [number, number] 
  {
  	return [this.heightMin, this.heightMax];
  }

  set elevationRange(value: [number, number]) 
  {
  	this.heightMin = value[0];
  	this.heightMax = value[1];
  }

  getUniform<K extends keyof IPointCloudMaterialUniforms>(
  	name: K,
  ): IPointCloudMaterialUniforms[K]['value'] 
  {
  	return this.uniforms === undefined ? (undefined as any) : this.uniforms[name].value;
  }

  setUniform<K extends keyof IPointCloudMaterialUniforms>(
  	name: K,
  	value: IPointCloudMaterialUniforms[K]['value'],
  ): void 
  {
  	if (this.uniforms === undefined) 
  	{
  		return;
  	}

  	const uObj = this.uniforms[name];

  	if (uObj.type === 'c') 
  	{
  		(uObj.value as Color).copy(value as Color);
  	}
  	else if (value !== uObj.value) 
  	{
  		uObj.value = value;
  	}
  }

  updateMaterial(
  	octree: PointCloudOctree,
  	visibleNodes: PointCloudOctreeNode[],
  	camera: Camera,
  	renderer: WebGLRenderer,
  ): void 
  {
	this.useLogDepth = renderer.capabilities.logarithmicDepthBuffer;

  	if (camera.type === PERSPECTIVE_CAMERA) 
  	{
  		this.useOrthographicCamera = false;
  		this.fov = (camera as PerspectiveCamera).fov * (Math.PI / 180);
		this.far = (camera as PerspectiveCamera).far;
  	}
  	else // ORTHOGRAPHIC
  	{
   		const orthoCamera = (camera as OrthographicCamera);
  		this.useOrthographicCamera = true;
  		this.orthoWidth = (orthoCamera.right - orthoCamera.left) / orthoCamera.zoom;
  		this.orthoHeight = (orthoCamera.top - orthoCamera.bottom) / orthoCamera.zoom;
  		this.fov = Math.PI / 2; // will result in slope = 1 in the shader
		this.far = (camera as OrthographicCamera).far;
  	}
  	const renderTarget = renderer.getRenderTarget();
  	if (renderTarget !== null && renderTarget instanceof WebGLRenderTarget) 
  	{
  		this.screenWidth = renderTarget.width;
  		this.screenHeight = renderTarget.height;
  	}
  	else 
  	{
  		// The drawing buffer, not the canvas's CSS box: no DOM read (works on
  		// an OffscreenCanvas) and exact under any pixel ratio.
  		const size = renderer.getDrawingBufferSize(PointCloudMaterial.helperVec2);
  		this.screenWidth = size.x;
  		this.screenHeight = size.y;
  	}


  	const maxScale = Math.max(octree.scale.x, octree.scale.y, octree.scale.z);
  	this.spacing = octree.pcoGeometry.spacing * maxScale;
  	this.octreeSize = octree.pcoGeometry.boundingBox.getSize(PointCloudMaterial.helperVec3).x;
	const view = (camera as any).view;
	if (view?.enabled) {
		this.viewScale = view.fullWidth / view.width;
	} else {
		this.viewScale = 1.0;
	}

  	if (
  		this.pointSizeType === PointSizeType.ADAPTIVE ||
	  this.pointColorType === PointColorType.LOD
  	) 
  	{
  		this.updateVisibilityTextureData(visibleNodes);
  	}
  	else 
  	{
  		// Texels of another frame would give stale per-node indices.
  		this.visibleNodeTextureOffsets.clear();
  	}
  }

  private updateVisibilityTextureData(nodes: PointCloudOctreeNode[])
  {
  	nodes.sort(byLevelAndIndex);

  	const rows = visibilityTextureRows(nodes.length);
  	if (!this.visibleNodesTexture || (this.visibleNodesTexture.image as { height: number }).height < rows)
  	{
  		this.allocateVisibleNodesTexture(rows);
  	}
  	packVisibleNodes(nodes, this.visibleNodesData, this.visibleNodeTextureOffsets);

  	if (this.visibleNodesTexture)
  	{
  		this.visibleNodesTexture.needsUpdate = true;
  	}
  }

  /** (Re)creates the visible-nodes texture with `rows` rows: three cannot resize an uploaded texture. */
  private allocateVisibleNodesTexture(rows: number): void
  {
  	this.visibleNodesTexture?.dispose();
  	const tex = generateDataTexture(VISIBLE_NODES_TEXTURE_WIDTH, rows, new Color(0xffffff));
  	tex.minFilter = NearestFilter;
  	tex.magFilter = NearestFilter;
  	this.visibleNodesTexture = tex;
  	this.visibleNodesData = (tex.image as { data: Uint8Array }).data;
  	this.setUniform('visibleNodes', tex);
  }

  static makeOnBeforeRender(
  	octree: PointCloudOctree,
  	node: PointCloudOctreeNode,
  	pcIndex?: number,
  ) 
  {
  	return (
  		_renderer: WebGLRenderer,
  		_scene: Scene,
  		_camera: Camera,
  		geometry: BufferGeometry,
  		material: Material,
  	) => 
  	{
			if (material instanceof PointCloudMaterial) {
				const materialUniforms = material.uniforms;

				materialUniforms.level.value = node.level;
				materialUniforms.isLeafNode.value = node.isLeafNode;

				const vnStart = material.visibleNodeTextureOffsets.get(node.name);
				if (vnStart !== undefined)
				{
					materialUniforms.vnStart.value = vnStart;
				}
				materialUniforms.hasOverlayScalar.value = geometry.getAttribute('overlayScalar') ? 1 : 0;
				materialUniforms.hasOverlayEigen.value = geometry.getAttribute('overlayEigen') ? 1 : 0;
				materialUniforms.hasNormal.value = geometry.getAttribute('normal') ? 1 : 0;
				materialUniforms.hasRgba.value = geometry.getAttribute('rgba') ? 1 : 0;
				// The node's scene position is its box min relative to the cloud
				// (potree.ts: matrixWorld = cloud.matrixWorld * node.matrix).
				materialUniforms.nodeOffset.value.copy(node.sceneNode.position);
				const result = material.nodeResultStates.get(node.name);
				materialUniforms.nodeResultState.value = result ? result.state : material.defaultNodeResultState;
				materialUniforms.resultAge.value = result ? performance.now() - result.since : 1e9;

				// visibleNodes is sorted in texture order when the texture is in
				// use, so a node's texel is its index (indexOf is O(nodes) per draw).
				materialUniforms.pcIndex.value =
				pcIndex !== undefined ? pcIndex : vnStart !== undefined ? vnStart : octree.visibleNodes.indexOf(node);

				// Remove the cast to any after updating to Three.JS >= r113
				(material as RawShaderMaterial).uniformsNeedUpdate = true;
			}
		};
  }
}

function makeUniform<T>(type: string, value: T): IUniform<T> 
{
	return {type: type, value: value};
}

function getValid<T>(a: T | undefined, b: T): T 
{
	return a === undefined ? b : a;
}

// tslint:disable:no-invalid-this
function uniform<K extends keyof IPointCloudMaterialUniforms>(
	uniformName: K,
	requireSrcUpdate: boolean = false,
): PropertyDecorator 
{
	return (target: Object, propertyKey: string | symbol): void => 
	{
		Object.defineProperty(target, propertyKey, {
			get: function() 
			{
				return this.getUniform(uniformName);
			},
			set: function(value: any) 
			{
				if (value !== this.getUniform(uniformName)) 
				{
					this.setUniform(uniformName, value);
					if (requireSrcUpdate) 
					{
						this.updateShaderSource();
					}
				}
			}
		});
	};
}

function requiresShaderUpdate() 
{
	return (target: Object, propertyKey: string | symbol): void => 
	{
		const fieldName = `_${propertyKey.toString()}`;

		Object.defineProperty(target, propertyKey, {
			get: function() 
			{
				return this[fieldName];
			},
			set: function(value: any) 
			{
				if (value !== this[fieldName]) 
				{
					this[fieldName] = value;
					this.updateShaderSource();
				}
			}
		});
	};
}
