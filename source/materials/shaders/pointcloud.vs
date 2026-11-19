precision highp float;
precision highp int;

#define max_clip_boxes 30   // Maximum number of clipping boxes
#define max_clip_spheres 30 // Maximum number of clipping spheres

// Input Attributes
in vec3 position;
in vec3 normal;
in float intensity;
in float classification;
in float returnNumber;
in float numberOfReturns;
in float pointSourceID;
in float gpsTime;
in float overlayScalar;
// Covariance eigenvalues l1 >= l2 >= l3 and the normal's z (live Select Field).
in vec4 overlayEigen;
// Per-point overlays written by processing results (defaults: visible, no tint).
in float overlayVisible;
in vec4 overlayRGBA;
in vec4 indices;


// Uniforms
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;

uniform float pcIndex;

uniform float screenWidth;
uniform float screenHeight;
uniform float fov;
uniform float spacing;
uniform float viewScale;

uniform bool useOrthographicCamera;
uniform float orthoWidth;
uniform float orthoHeight;

#if defined use_clip_box
	uniform mat4 clipBoxes[max_clip_boxes]; // Clipping box transforms
#endif

#if defined use_clip_sphere
	uniform vec4 clipSpheres[max_clip_spheres]; // Clipping spheres: xyz = center, w = radius
	uniform float clipSphereCount;
#endif

uniform float heightMin;
uniform float heightMax;
uniform float size; // Base pixel size
uniform float minSize; // Minimum point size
uniform float maxSize; // Maximum point size
uniform float octreeSize;
uniform vec3 bbSize;
uniform vec3 uColor;
uniform float opacity;
uniform float clipBoxCount;
uniform float level;
uniform float vnStart;
uniform bool isLeafNode;

uniform float filterByNormalThreshold;
uniform vec2 intensityRange;
uniform float opacityAttenuation;
uniform float intensityGamma;
uniform float intensityContrast;
uniform float intensityBrightness;
uniform float rgbGamma;
uniform float rgbContrast;
uniform float rgbBrightness;
uniform float transition;
uniform float wRGB;
uniform float wIntensity;
uniform float wElevation;
uniform float wClassification;
uniform float wReturnNumber;
uniform float wSourceID;

uniform sampler2D visibleNodes;
uniform sampler2D gradient;
uniform sampler2D classificationLUT;
uniform sampler2D depthMap;

uniform float overlayMin;
uniform float overlayMax;
// Scalar range filter: a uniform update per slider move, no per-point data.
uniform vec2 scalarFilterRange;
uniform float scalarFilterEnabled;
uniform float hasOverlayScalar;

// ── Live style (uniforms only: a slider move never recompiles) ──────────────
// Every test below has a CPU twin (app/src/style/styleCpuTwin.ts) that must
// agree bit for bit: only exactly rounded operations (+, -, *, compares) feed
// a hide decision, never division or a transcendental function.
uniform int classHidden[8];         // 256-bit mask of hidden classes, every colour mode
uniform int attrFilterAttr[4];      // 0 off, 1 intensity, 2 return number, 3 number of returns,
                                    // 4 point source id, 5 gps time, 6 classification
uniform vec2 attrFilterRange[4];    // keep [x, y] in the attribute's stored units
uniform float clipAabbMode;         // 0 off, 1 keep inside, 2 keep outside (PassThrough / height range)
uniform vec3 clipAabbMin;           // cloud frame (position + nodeOffset)
uniform vec3 clipAabbMax;
uniform vec3 nodeOffset;            // per node: its box min in the cloud frame
uniform int scalarSource;           // 0 overlayScalar, 1 covariance feature, 2 slope (deg), 3 verticality
uniform int featureIndex;           // 0 linearity .. 7 verticality (ScalarFieldId order)
uniform vec2 scalarFilterDomain;    // scalarFilterRange mapped to the exact test's domain
uniform float normalFilterEnabled;  // slope filter over the stored normal
uniform vec2 normalFilterAbsNz;     // keep |n.z| in [x, y] (= slope in [acos y, acos x])
uniform float hasOverlayEigen;      // per node
uniform float hasNormal;            // per node
uniform float hasRgba;              // per node
uniform float rgbAdjust;            // 1: apply rgbGamma/Brightness/Contrast
uniform float nodeResultState;      // per node: 0 untracked, 1 pending, 2 preview, 3 exact
uniform float resultAge;            // per node: ms since its result arrived
uniform float previewMarks;         // 1: desaturate preview nodes
uniform float reducedMotion;        // 1: no shimmer
uniform float styleTime;            // seconds, drives the preview shimmer

// One texel per visible node, in rows of textureSize(visibleNodes).x texels.
vec4 visibleNodeTexel(int i) {
	int width = textureSize(visibleNodes, 0).x;
	return texelFetch(visibleNodes, ivec2(i % width, i / width), 0);
}

#ifdef highlight_point
	uniform vec3 highlightedPointCoordinate;
	uniform bool enablePointHighlighting;
	uniform float highlightedPointScale;
#endif

#ifdef new_format
	in vec4 rgba;
	out vec4 vColor;
#else
	in vec3 color;
	out vec3 vColor;
#endif

#if !defined(color_type_point_index)
	out float vOpacity;
#endif

#if defined(weighted_splats)
	out float vLinearDepth;
#endif



out vec3 vViewPosition;

#if defined(weighted_splats) || defined(paraboloid_point_shape)
	out float vRadius;
#endif

#if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0)
	out vec3 vNormal;
#endif

#ifdef highlight_point
	out float vHighlight;
#endif


// OCTREE LOD FUNCTIONS
#if (defined(adaptive_point_size) || defined(color_type_lod)) && defined(tree_type_octree)

// Returns count of bits set up to a given index
int numberOfOnes(int number, int index) {
	int numOnes = 0;
	int tmp = 128;
	for (int i = 7; i >= 0; i--) {
		if (number >= tmp) {
			number -= tmp;
			if (i <= index) { numOnes++; }
		}
		tmp /= 2;
	}
	return numOnes;
}

// Checks if bit at specific index is set
bool isBitSet(int number, int index){
	return ((number >> index) & 1) != 0;
}

// Computes level-of-detail based on octree visibility
float getLOD() {
	vec3 offset = vec3(0.0);
	int iOffset = int(vnStart);
	float depth = level;

	for (float i = 0.0; i <= 30.0; i++) {
		float nodeSizeAtLevel = octreeSize / pow(2.0, i + level);
		vec3 index3d = floor((position - offset) / nodeSizeAtLevel + 0.5);
		int index = int(round(4.0 * index3d.x + 2.0 * index3d.y + index3d.z));
		
		vec4 value = visibleNodeTexel(iOffset);
		int mask = int(round(value.r * 255.0));

		if (isBitSet(mask, index)) {
			int advanceG = int(round(value.g * 255.0)) * 256;
			int advanceB = int(round(value.b * 255.0));
			int advanceChild = numberOfOnes(mask, index - 1);
			int advance = advanceG + advanceB + advanceChild;
			iOffset += advance;
			depth++;
		} else {
			return depth;
		}
		offset += (vec3(1.0) * nodeSizeAtLevel * 0.5) * index3d;
	}
	return depth;
}

float getPointSizeAttenuation() {
	return 0.5 * pow(2.0, getLOD());
}

#endif


// KD-TREE LOD FUNCTIONS

#if (defined(adaptive_point_size) || defined(color_type_lod)) && defined(tree_type_kdtree)

float getLOD() {
	vec3 offset = vec3(0.0);
	float intOffset = 0.0;
	float depth = 0.0;
	vec3 size_ = bbSize;
	vec3 pos = position;
		
	for (float i = 0.0; i <= 1000.0; i++) {
		vec4 value = visibleNodeTexel(int(intOffset));
		int children = int(value.r * 255.0);
		float next = value.g * 255.0;
		int split = int(value.b * 255.0);
		
		if (next == 0.0) { return depth; }
		
		vec3 splitv = (split == 1) ? vec3(1.0, 0.0, 0.0) :
					 (split == 2) ? vec3(0.0, 1.0, 0.0) :
					 (split == 4) ? vec3(0.0, 0.0, 1.0) : vec3(0.0);
		
		intOffset += next;
		
		float factor = length((pos * splitv) / size_);
		if (factor < 0.5) {
			if (children == 0 || children == 2) { return depth; }
		} else {
			pos -= size_ * splitv * 0.5;
			if (children == 0 || children == 1) { return depth; }
			if (children == 3) { intOffset += 1.0; }
		}
		size_ *= ((1.0 - (splitv + 1.0) / 2.0) + 0.5);
		depth++;
	}
	return depth;	
}

float getPointSizeAttenuation() {
	return 0.5 * pow(1.3, getLOD());
}

#endif


// COLOR/BRIGHTNESS FUNCTIONS

float getContrastFactor(float contrast) {
	return 1.0158730158730156 * (contrast + 1.0) / (1.0158730158730156 - contrast);
}

// The point's own colour (Potree 2 "new format" clouds carry it in rgba).
vec3 rawRGB() {
	#ifdef new_format
		return rgba.rgb;
	#else
		return color;
	#endif
}

// Adjusts RGB values with gamma, contrast and brightness factors (a uniform
// switch, so dragging a slider never recompiles the shader).
vec3 getRGB() {
	vec3 rgb = rawRGB();
	if (rgbAdjust > 0.5) {
		rgb = pow(max(rgb, vec3(0.0)), vec3(rgbGamma));
		rgb += rgbBrightness;
		rgb = (rgb - 0.5) * getContrastFactor(rgbContrast) + 0.5;
		rgb = clamp(rgb, 0.0, 1.0);
	}
	return rgb;
}

// Adjusts intensity value based on settings
float getIntensity() {
	float w = (intensity - intensityRange.x) / (intensityRange.y - intensityRange.x);
	w = pow(w, intensityGamma) + intensityBrightness;
	w = (w - 0.5) * getContrastFactor(intensityContrast) + 0.5;
	return clamp(w, 0.0, 1.0);
}

// Maps elevation to a gradient color
vec3 getElevation() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	float w = (world.z - heightMin) / (heightMax - heightMin);
	return texture(gradient, vec2(w, 1.0 - w)).rgb;
}

// Gets classification color and transparency from texture LUT
vec4 getClassification() {
	vec2 uv = vec2(classification / 255.0, 0.5);
	return texture(classificationLUT, uv);
}

// Returns color based on number of returns and return number
vec3 getReturnNumber() {
	return (numberOfReturns == 1.0) ? vec3(1.0, 1.0, 0.0) :
		   (returnNumber == 1.0)    ? vec3(1.0, 0.0, 0.0) :
		   (returnNumber == numberOfReturns) ? vec3(0.0, 0.0, 1.0) :
										vec3(0.0, 1.0, 0.0);
}

bool isNaNBits(float v) {
	uint bits = floatBitsToUint(v);
	return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

// A range test hides every value outside [lo, hi], NaN included (as
// scalar-filter.ts does on the CPU). A NaN fails every comparison, so the test
// is !(lo <= v <= hi), never (v < lo || v > hi); the bit test catches drivers
// that fold NaN comparisons away.
bool rangeHides(float v, vec2 range) {
	return isNaNBits(v) || !(v >= range.x && v <= range.y);
}

bool scalarFilterHides(float v) {
	return rangeHides(v, scalarFilterRange);
}

// Covariance features (Weinmann et al. 2017), ScalarFieldId order; e = (l1, l2, l3, n.z).
float featureValue(vec4 e, int f) {
	float l1 = e.x, l2 = e.y, l3 = e.z;
	float sumEv = (l1 + l2) + l3;
	if (f == 0) return (l1 - l2) / l1;
	if (f == 1) return (l2 - l3) / l1;
	if (f == 2) return l3 / l1;
	if (f == 3) return pow(max((l1 * l2) * l3, 0.0), 1.0 / 3.0);
	if (f == 4) return (l1 - l3) / l1;
	if (f == 5) {
		float h = 0.0;
		vec3 n = vec3(l1, l2, l3) / sumEv;
		if (n.x > 1e-10) h -= n.x * log(n.x);
		if (n.y > 1e-10) h -= n.y * log(n.y);
		if (n.z > 1e-10) h -= n.z * log(n.z);
		return h;
	}
	if (f == 6) return l3 / sumEv;
	return 1.0 - abs(e.w);
}

// The value the scalar colour mode and the scalar filter use.
float styleScalar() {
	if (scalarSource == 1) return featureValue(overlayEigen, featureIndex);
	if (scalarSource == 2) return degrees(acos(clamp(abs(normal.z), 0.0, 1.0)));
	if (scalarSource == 3) return 1.0 - abs(normal.z);
	return overlayScalar;
}

// Per node: whether this node carries the scalar's inputs.
float styleHasScalar() {
	if (scalarSource == 1) return hasOverlayEigen;
	if (scalarSource >= 2) return hasNormal;
	return hasOverlayScalar;
}

// Ratio features are tested as lo * den <= num <= hi * den (den > 0): exact
// operations only, so the CPU twin agrees bit for bit with the screen.
bool ratioHides(float num, float den) {
	return isNaNBits(num) || isNaNBits(den) ||
		!(scalarFilterDomain.x * den <= num && num <= scalarFilterDomain.y * den);
}

// The scalar range filter over styleScalar(), in its exact form.
bool styleScalarHides() {
	if (scalarSource == 1) {
		vec4 e = overlayEigen;
		float l1 = e.x, l2 = e.y, l3 = e.z;
		if (featureIndex == 0) return ratioHides(l1 - l2, l1);
		if (featureIndex == 1) return ratioHides(l2 - l3, l1);
		if (featureIndex == 2) return ratioHides(l3, l1);
		if (featureIndex == 3) return rangeHides((l1 * l2) * l3, scalarFilterDomain);
		if (featureIndex == 4) return ratioHides(l1 - l3, l1);
		if (featureIndex == 6) return ratioHides(l3, (l1 + l2) + l3);
		if (featureIndex == 7) return rangeHides(1.0 - abs(e.w), scalarFilterRange);
		return rangeHides(featureValue(e, featureIndex), scalarFilterRange);
	}
	if (scalarSource == 2) return rangeHides(abs(normal.z), scalarFilterDomain);
	if (scalarSource == 3) return rangeHides(1.0 - abs(normal.z), scalarFilterRange);
	return rangeHides(overlayScalar, scalarFilterRange);
}

bool classIsHidden() {
	int c = clamp(int(classification + 0.5), 0, 255);
	return ((classHidden[c >> 5] >> (c & 31)) & 1) != 0;
}

float styleAttrValue(int a) {
	if (a == 1) return intensity;
	if (a == 2) return returnNumber;
	if (a == 3) return numberOfReturns;
	if (a == 4) return pointSourceID;
	if (a == 5) return gpsTime;
	return classification;
}

bool attrFiltersHide() {
	for (int k = 0; k < 4; k++) {
		int a = attrFilterAttr[k];
		if (a != 0 && rangeHides(styleAttrValue(a), attrFilterRange[k])) return true;
	}
	return false;
}

bool clipAabbHides() {
	if (clipAabbMode < 0.5) return false;
	vec3 p = position + nodeOffset;
	bool inside = all(greaterThanEqual(p, clipAabbMin)) && all(lessThanEqual(p, clipAabbMax));
	return clipAabbMode < 1.5 ? !inside : inside;
}

// What a node without its result yet draws: its own colour, never the
// gradient's low end (a pending node would otherwise read as "value 0").
vec3 baseColor() {
	return hasRgba > 0.5 ? rawRGB() : vec3(0.62);
}

// Maps the style scalar to the gradient, honouring the node's result state.
vec3 getOverlayScalar() {
	bool pending = nodeResultState > 0.5 && nodeResultState < 1.5;
	if (pending || styleHasScalar() < 0.5) return baseColor();
	float w = clamp((styleScalar() - overlayMin) / max(overlayMax - overlayMin, 0.0001), 0.0, 1.0);
	vec3 c = texture(gradient, vec2(w, 1.0 - w)).rgb;
	if (nodeResultState > 1.5 && nodeResultState < 2.5 && previewMarks > 0.5) {
		// Preview: desaturated, with a slow 1.6 s shimmer unless motion is reduced.
		float shimmer = reducedMotion > 0.5 ? 0.0 : 0.08 * sin(styleTime * 3.92699 + position.x * 0.7 + position.y * 0.5);
		float luma = dot(c, vec3(0.299, 0.587, 0.114));
		c = mix(c, vec3(luma), 0.45) + shimmer;
	}
	if (nodeResultState > 2.5 && resultAge < 150.0) {
		c = mix(baseColor(), c, clamp(resultAge / 150.0, 0.0, 1.0));
	}
	return c;
}

vec3 getSourceID() {
	float w = mod(pointSourceID, 10.0) / 10.0;
	return texture(gradient, vec2(w, 1.0 - w)).rgb;
}

#ifndef new_format
// Combines multiple color sources into one composite color
vec3 getCompositeColor() {
	vec3 c = wRGB * getRGB();
	float w = wRGB;
	c += wIntensity * getIntensity() * vec3(1.0);
	w += wIntensity;
	c += wElevation * getElevation();
	w += wElevation;
	c += wReturnNumber * getReturnNumber();
	w += wReturnNumber;
	c += wSourceID * getSourceID();
	w += wSourceID;

	vec4 cl = wClassification * getClassification();
	c += cl.a * cl.rgb;
	w += wClassification * cl.a;

	c /= w;
	if (w == 0.0) {
		gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
	}
	return c;
}
#endif

#ifdef new_format
// sRGB conversion functions
vec4 fromLinear(vec4 linearRGB) {
	bvec4 cutoff = lessThan(linearRGB, vec4(0.0031308));
	return mix(linearRGB * vec4(12.92), vec4(1.055) * pow(linearRGB, vec4(1.0/2.4)) - vec4(0.055), cutoff);
} 
vec4 toLinear(vec4 sRGB) {
	bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
	return mix(sRGB/vec4(12.92), pow((sRGB + vec4(0.055))/vec4(1.055), vec4(2.4)), cutoff);
}
#else
vec3 fromLinear(vec3 linearRGB) {
	bvec3 cutoff = lessThan(linearRGB, vec3(0.0031308));
	return mix(linearRGB * vec3(12.92), vec3(1.055) * pow(linearRGB, vec3(1.0/2.4)) - vec3(0.055), cutoff);
}
vec3 toLinear(vec3 sRGB) {
	bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
	return mix(sRGB/vec3(12.92), pow((sRGB + vec3(0.055))/vec3(1.055), vec3(2.4)), cutoff);
}
#endif

void main() {
	// Compute model-view position and projected position
	vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * mvPosition;
	vViewPosition = mvPosition.xyz;

	#if defined weighted_splats
		vLinearDepth = gl_Position.w;
	#endif

	#if defined(color_type_phong) && (MAX_POINT_LIGHTS > 0 || MAX_DIR_LIGHTS > 0)
		vNormal = normalize(normalMatrix * normal);
	#endif

	// EDL alpha (log2(linearDepth)) is written in the fragment shader so it can
	// account for per-fragment depth adjustments (e.g. paraboloid point shape).


	// POINT SIZE COMPUTATION
	float tanHalfFOV = tan(fov * 0.5);
	float projFactor = -0.5 * screenHeight / (tanHalfFOV * mvPosition.z);
	// Scale compensation based on transformation difference 
	float scale = length(modelViewMatrix * vec4(0, 0, 0, 1) - modelViewMatrix * vec4(spacing, 0, 0, 1)) / spacing;
	projFactor *= scale;
	
	float pointSize = 1.0;
	#if defined fixed_point_size
		pointSize = size;
	#elif defined attenuated_point_size
		pointSize = useOrthographicCamera ? size : size * spacing * projFactor;
	#elif defined adaptive_point_size
		float worldSpaceSize = 2.0 * size * spacing / getPointSizeAttenuation();
		pointSize = useOrthographicCamera ? (worldSpaceSize / orthoWidth) * screenWidth : worldSpaceSize * projFactor;
	#endif

	pointSize = clamp(pointSize, minSize, maxSize);
	#if defined(weighted_splats) || defined(paraboloid_point_shape)
		vRadius = pointSize / projFactor;
	#endif
	
	pointSize *= viewScale;
	gl_PointSize = pointSize;


	// HIGHLIGHTING
	#ifdef highlight_point
		vec4 mPosition = modelMatrix * vec4(position, 1.0);
		if (enablePointHighlighting &&
			abs(mPosition.x - highlightedPointCoordinate.x) < 0.0001 &&
			abs(mPosition.y - highlightedPointCoordinate.y) < 0.0001 &&
			abs(mPosition.z - highlightedPointCoordinate.z) < 0.0001) {
			vHighlight = 1.0;
			gl_PointSize = pointSize * highlightedPointScale;
		} else {
			vHighlight = 0.0;
		}
	#endif


	// OPACITY
	#ifndef color_type_point_index
		#ifdef attenuated_opacity
			vOpacity = opacity * exp(-length(-mvPosition.xyz) / opacityAttenuation);
		#else
			vOpacity = opacity;
		#endif
	#endif


	// NORMAL FILTERING
	#ifdef use_filter_by_normal
		if(abs((modelViewMatrix * vec4(normal, 0.0)).z) > filterByNormalThreshold) {
			gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // Cull point
		}
	#endif


	// POINT COLOR SELECTION
	// Potree 2 clouds used to always show rgba, ignoring pointColorType, so no
	// other colour mode (classification, elevation, scalar overlays...) ever
	// reached the screen. rgba is now only the default.
	#ifdef new_format
		#define POTREE_SET_COLOR(c) vColor = vec4((c), rgba.a)
		vColor = rgba;
	#else
		#define POTREE_SET_COLOR(c) vColor = (c)
	#endif

	#if defined color_type_rgb
		POTREE_SET_COLOR(getRGB());
	#elif defined color_type_height
		POTREE_SET_COLOR(getElevation());
	#elif defined color_type_rgb_height
		vec3 cHeight = getElevation();
		POTREE_SET_COLOR(mix(getRGB(), cHeight, transition));
	#elif defined color_type_depth
		float linearDepth = -mvPosition.z;
		float expDepth = (gl_Position.z / gl_Position.w) * 0.5 + 0.5;
		POTREE_SET_COLOR(vec3(linearDepth, expDepth, 0.0));
	#elif defined color_type_intensity
		float w = getIntensity();
		POTREE_SET_COLOR(vec3(w));
	#elif defined color_type_intensity_gradient
		float w = getIntensity();
		POTREE_SET_COLOR(texture(gradient, vec2(w, 1.0 - w)).rgb);
	#elif defined color_type_color
		POTREE_SET_COLOR(uColor);
	#elif defined color_type_lod
		float w = getLOD() / 10.0;
		POTREE_SET_COLOR(texture(gradient, vec2(w, 1.0 - w)).rgb);
	#elif defined color_type_point_index
		POTREE_SET_COLOR(indices.rgb);
	#elif defined color_type_classification
		vec4 cl = getClassification();
		POTREE_SET_COLOR(cl.rgb);
	#elif defined color_type_return_number
		POTREE_SET_COLOR(getReturnNumber());
	#elif defined color_type_source
		POTREE_SET_COLOR(getSourceID());
	#elif defined color_type_normal
		POTREE_SET_COLOR((modelMatrix * vec4(normal, 0.0)).xyz);
	#elif defined color_type_phong
		POTREE_SET_COLOR(color);
	#elif defined color_type_overlay_scalar
		POTREE_SET_COLOR(getOverlayScalar());
	#elif defined color_type_composite
		POTREE_SET_COLOR(getCompositeColor());
	#endif
	
	// Processing overlays: hidden points (e.g. removed by a filter) are culled,
	// tinted points (e.g. cluster labels) override the colour mode.
	if (overlayVisible < 0.5) {
		gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
		return;
	}
	// The live style's hide tests (CPU twin: app/src/style/styleCpuTwin.ts).
	// A pending node (no result yet) is never filtered by its missing value.
	bool resultPending = nodeResultState > 0.5 && nodeResultState < 1.5;
	if (scalarFilterEnabled > 0.5 && styleHasScalar() > 0.5 && !resultPending && styleScalarHides()) {
		gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
		return;
	}
	if (classIsHidden() || attrFiltersHide() || clipAabbHides() ||
		(normalFilterEnabled > 0.5 && hasNormal > 0.5 && rangeHides(abs(normal.z), normalFilterAbsNz))) {
		gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
		return;
	}
	if (overlayRGBA.a > 0.0) {
		vColor.rgb = overlayRGBA.rgb;
	}

	#if !defined color_type_composite && defined color_type_classification
		if (getClassification().a == 0.0) {
			gl_Position = vec4(100.0, 100.0, 100.0, 0.0); // Cull point if classification alpha is zero
			return;
		}
	#endif


	// CLIPPING
	#if defined use_clip_box || defined use_clip_sphere
		bool insideAny = false;

		#if defined use_clip_box
			for (int i = 0; i < max_clip_boxes; i++) {
				if (i == int(clipBoxCount)) break;
				vec4 clipPosition = clipBoxes[i] * modelMatrix * vec4(position, 1.0);
				bool inside = abs(clipPosition.x) <= 0.5 && abs(clipPosition.y) <= 0.5 && abs(clipPosition.z) <= 0.5;
				insideAny = insideAny || inside;
			}
		#endif

		#if defined use_clip_sphere
			vec4 modelPos = modelMatrix * vec4(position, 1.0);
			for (int i = 0; i < max_clip_spheres; i++) {
				if (i == int(clipSphereCount)) break;
				float dist = distance(modelPos.xyz, clipSpheres[i].xyz);
				insideAny = insideAny || (dist <= clipSpheres[i].w);
			}
		#endif

		#if defined clip_outside
			if (!insideAny) { gl_Position = vec4(1000.0); } // Cull if outside any clip volume
		#elif defined clip_inside
			if (insideAny) { gl_Position = vec4(1000.0); } // Cull if inside any clip volume
		#elif defined clip_highlight_inside && !defined(color_type_depth)
			if (!insideAny) { /* additional processing if needed */ }
		#endif

		#if defined clip_highlight_inside
			if (insideAny) { vColor.r += 0.5; }
		#endif
	#endif


	// COLOR ENCODING ADJUSTMENTS
	#if defined(output_color_encoding_sRGB) && defined(input_color_encoding_linear) && !defined(color_type_point_index)
		vColor = toLinear(vColor);
	#endif

	#if defined(output_color_encoding_linear) && defined(input_color_encoding_sRGB) && !defined(color_type_point_index)
		vColor = fromLinear(vColor);
	#endif
}
