// =====================================================================
//  《3D 脸谱 · 变脸 · 3D Opera Masks》— 起始文件 / starter
//  PART A（下面）是面部识别引擎，不要修改。/ PART A (below) is the face engine: don't change it.
//  PART B 只留下 5 个 ★ 标记的函数：每一步让 Qwen 改写其中一个。
//  PART B leaves 5 functions marked with ★: at each step, Qwen rewrites one of them.
// =====================================================================


// ---------------------------------------------------------------------
//  PART A — 面部识别引擎（三个项目完全相同，不要修改）
//  PART A — face engine (identical in all three projects, don't change it)
// ---------------------------------------------------------------------
const MP_VERSION = "0.10.21";
// MediaPipe 的来源：先用项目文件夹里的 lib/mediapipe（不需要网络），再试国内镜像 npmmirror，最后 jsDelivr
// MediaPipe sources: first the local lib/mediapipe folder (no internet needed), then npmmirror, then jsDelivr
const MP_URLS = [
  "./lib/mediapipe",
  `https://registry.npmmirror.com/@mediapipe/tasks-vision/${MP_VERSION}/files`,
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`,
];
// 面部模型：先找项目文件夹里的 face_landmarker.task，没有再从 Google 下载
const MODEL_URLS = [
  "face_landmarker.task",
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
];

// 常用的面部关键点编号（一共 478 个点）。画面是镜像的：你的左眼显示在画面左边。
// Face landmark numbers (478 in total). The picture is mirrored: your left eye is on the left of the screen.
const FACE = {
  FOREHEAD: 10, CHIN: 152, NOSE_TIP: 1,
  LEFT_CHEEK: 454, RIGHT_CHEEK: 234,           // 脸的左右边缘 / left and right edge of the face
  UPPER_LIP: 13, LOWER_LIP: 14, MOUTH_LEFT: 291, MOUTH_RIGHT: 61,
  LEFT_EYE_OUTER: 263, LEFT_EYE_INNER: 362, LEFT_EYE_TOP: 386, LEFT_EYE_BOTTOM: 374,
  RIGHT_EYE_OUTER: 33, RIGHT_EYE_INNER: 133, RIGHT_EYE_TOP: 159, RIGHT_EYE_BOTTOM: 145,
  LEFT_IRIS: 473, RIGHT_IRIS: 468,             // 瞳孔中心 / centre of the pupils
};
// 脸的轮廓（按顺序围成一圈）/ the outline of the face (in order, all the way round)
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
// 眼睛的一圈 / the ring around each eye
const LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
// 眉毛（从鼻子那边到外侧）/ eyebrows (from the nose side to the outside)
const LEFT_BROW = [336, 296, 334, 293, 300];
const RIGHT_BROW = [107, 66, 105, 63, 70];
// 嘴唇外圈 / outer ring of the lips
const LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];

let video;                 // 摄像头画面 / the webcam
let faceLandmarker = null; // MediaPipe 面部模型 / the face model
let faces = [];            // 每一帧：看到的所有脸（最多 2 张）。每张脸是一个对象：
                           // every frame: the faces the camera sees (max 2). Each face is an object:
                           //   face.points — 478 个点 {x, y, z}（画布像素坐标，已经镜像；z 越大，离摄像头越近）
                           //                 478 points (canvas pixels, mirrored; bigger z = closer to the camera)
                           //   face.blend  — 表情分数 0~1，例如 face.blend.jawOpen（张嘴）、face.blend.browInnerUp（扬眉）
                           //                 expression scores 0–1, e.g. face.blend.jawOpen, face.blend.browInnerUp
let lastVideoTime = -1;
let started = false;
let showCamera = false;
let showMesh = false;

// 如果一个下载来源太慢，最多等这么久就换下一个 / if a download source is too slow, wait at most this long, then try the next
function withTimeout(promise, ms) {
  return Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms))]);
}

async function loadFaceLandmarker() {
  setStatus("正在加载面部模型… / Loading the face model…");
  for (const base of MP_URLS) {
    let vision, fileset;
    try {
      vision = await withTimeout(import(`${base}/vision_bundle.mjs`), 20000);
      fileset = await vision.FilesetResolver.forVisionTasks(`${base}/wasm`);
    } catch (e) {
      continue; // 这个来源打不开或太慢，换下一个 / this source fails or is too slow: next one
    }
    let tooSlow = false;
    for (const model of MODEL_URLS) {
      for (const delegate of ["GPU", "CPU"]) {
        try {
          faceLandmarker = await withTimeout(vision.FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: model, delegate },
            runningMode: "VIDEO",
            numFaces: 2,
            outputFaceBlendshapes: true,
          }), 45000);
          return;
        } catch (e) {
          if (e.message === "timeout" && delegate === "CPU") { tooSlow = true; break; } // 网络太慢 / network too slow
          // 失败就试下一个 / try the next one
        }
      }
      if (tooSlow) break;
    }
  }
  setStatus("⚠️ 面部模型加载失败：请检查网络，并把 face_landmarker.task 放进项目文件夹 / Face model failed to load: check the network and put face_landmarker.task in the project folder");
}

// 导入的文件找不到时，在状态栏里说清楚是哪个文件 / when an imported file can't be loaded, say which one
function mediaError(path) {
  setStatus("⚠️ 找不到文件 / File not found: " + path);
  console.error("File not found: " + path);
}

// 每一帧调用一次 / call once per frame
let cameraOk = null;
function updateFaces() {
  if (!faceLandmarker || !video) return;
  const ok = video.elt.readyState >= 2;
  if (ok !== cameraOk) {
    cameraOk = ok;
    setStatus(ok ? "看着摄像头 🙂 / Look at the camera" : "正在等待摄像头… / Waiting for the camera…");
  }
  if (!ok) return;
  if (video.elt.currentTime === lastVideoTime) return;
  lastVideoTime = video.elt.currentTime;

  const result = faceLandmarker.detectForVideo(video.elt, performance.now());
  const fit = videoFit();
  faces = result.faceLandmarks.map((pts, i) => {
    const blend = {};
    const cats = result.faceBlendshapes && result.faceBlendshapes[i] ? result.faceBlendshapes[i].categories : [];
    for (const c of cats) blend[c.categoryName] = c.score;
    return { points: pts.map((p) => ({ x: fit.x + (1 - p.x) * fit.w, y: fit.y + p.y * fit.h, z: -p.z * fit.w })), blend };
  });
}

// 摄像头画面怎样铺满画布而不变形（像照片一样裁掉多出来的边）
// how the camera image fills the canvas without being squashed (cropping the extra edges, like a photo)
function videoFit() {
  const el = video.elt;
  const vw = el.videoWidth || el.width || 640;
  const vh = el.videoHeight || el.height || 480;
  const s = max(width / vw, height / vh);
  return { x: (width - vw * s) / 2, y: (height - vh * s) / 2, w: vw * s, h: vh * s };
}

// 脸的宽度（像素），用来让大小跟着脸走 / face width in pixels, so sizes follow the face
function faceSize(face) {
  const a = face.points[FACE.LEFT_CHEEK], b = face.points[FACE.RIGHT_CHEEK];
  return dist(a.x, a.y, b.x, b.y);
}

// 画出脸上的点（学习和调试用，按 K 显示/隐藏）/ face mesh dots for learning and debugging (key K)
function drawFaceMesh() {
  push();
  noStroke();
  for (const face of faces) {
    fill(255, 255, 0, 160);
    for (let i = 0; i < 468; i += 3) circle(face.points[i].x, face.points[i].y, 2);
    fill(255, 60, 60);
    for (const i of [...FACE_OVAL, ...LEFT_EYE_RING, ...RIGHT_EYE_RING, ...LEFT_BROW, ...RIGHT_BROW, ...LIPS]) circle(face.points[i].x, face.points[i].y, 3);
    fill(0, 200, 255);
    if (face.points.length > 473) { circle(face.points[468].x, face.points[468].y, 6); circle(face.points[473].x, face.points[473].y, 6); }
  }
  pop();
}

function drawCamera() {
  const fit = videoFit();
  push();
  translate(width, 0);
  scale(-1, 1);
  tint(255, 90);
  image(video, fit.x, fit.y, fit.w, fit.h);
  pop();
}

function setStatus(msg) {
  select("#status").html(msg);
}

// 用浏览器检查 GLSL 有没有错误（没有错误就返回 ""）/ ask the browser whether the GLSL has an error ("" = no error)
function glslError(src) {
  const gl = document.createElement("canvas").getContext("webgl");
  if (!gl) return "";
  const s = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? "" : (gl.getShaderInfoLog(s) || "compile error").split("\n")[0];
}


// ---------------------------------------------------------------------
//  PART B — 作品：《3D 脸谱 · 变脸》（京剧脸谱、川剧变脸）
//  PART B — artwork: "3D Opera Masks · Face-Changing" (Peking opera, Sichuan opera)
// ---------------------------------------------------------------------

// 脸谱列表：每一张脸谱 = 一个 3D 模型文件（.obj）+ 一张贴图（.png / .jpg）。
// 想换脸谱或加新的脸谱，只要改这个列表（看课程里的“换一张脸谱”）。
// The mask list: every mask = one 3D model file (.obj) + one texture image (.png / .jpg).
// To change a mask or add a new one, only change this list (see "Change the mask" in the lesson).
//   size：     面具宽度 = 脸宽 × size / mask width = face width × size
//   x, y, z：  往右 / 往下 / 往前移动多少（单位：脸宽）/ move right / down / forward (in face widths)
//   turnX/Y/Z：额外转多少度 / extra rotation in degrees
//   flipY：    模型是“Y 轴朝上”的就写 true（大部分 .obj 文件都是）/ true if the model is "Y up" (most .obj files are)
const MASKS = [
  { name: "关羽 · 红脸 · 忠义 / Guan Yu · red · loyalty", model: "masks/face.obj", texture: "masks/guanyu.png",
    size: 1.05, x: 0, y: 0, z: 0, turnX: 0, turnY: 0, turnZ: 0, flipY: true },
  { name: "包拯 · 黑脸 · 公正 / Bao Zheng · black · justice", model: "masks/face.obj", texture: "masks/baozheng.png",
    size: 1.05, x: 0, y: 0, z: 0, turnX: 0, turnY: 0, turnZ: 0, flipY: true },
  { name: "曹操 · 白脸 · 奸诈 / Cao Cao · white · cunning", model: "masks/face.obj", texture: "masks/caocao.png",
    size: 1.05, x: 0, y: 0, z: 0, turnX: 0, turnY: 0, turnZ: 0, flipY: true },
  { name: "窦尔敦 · 蓝脸 · 刚强 / Dou Erdun · blue · bold", model: "masks/face.obj", texture: "masks/douerdun.png",
    size: 1.05, x: 0, y: 0, z: 0, turnX: 0, turnY: 0, turnZ: 0, flipY: true },
  { name: "孙悟空 · 金脸 · 神通 / Sun Wukong · gold · magic", model: "masks/face.obj", texture: "masks/wukong.png",
    size: 1.05, x: 0, y: 0, z: 0, turnX: 0, turnY: 0, turnZ: 0, flipY: true },
  // 从网上下载的傩戏面具（看课程）/ a Nuo opera mask downloaded from the internet (see the lesson):
  // { name: "傩面 · Nuo mask", model: "masks/nuo.obj", texture: "masks/nuo.jpg",
  //   size: 1.3, x: 0, y: 0, z: 0, turnX: 0, turnY: 0, turnZ: 0, flipY: true },
];

let maskIndex = 0;
let flash = 0;              // 变脸时的白光 / white flash when the mask changes
let lastExpr = [];          // 每张脸上一帧的表情 / last frame's expression of each face
let lastChange = -100;
let stage = null;           // 舞台灯光着色器 / stage light shader
let shaderError = "";
let calibrate = false;      // 校准模式（按 C）/ calibration mode (key C)

// 导入的声音 / imported sounds
let gong, cymbal, woodblock;
let reverb;

function setup() {
  createCanvas(windowWidth, windowHeight, WEBGL);
  pixelDensity(1);
  setCamera3D();
  makeShader();
  loadMedia();
  showMaskName();
  select("#start").mouseClicked(startExperience);
}

// 正交相机：3D 物体不会因为远近变大变小，画布坐标和脸上的点一一对应
// orthographic camera: no perspective, so canvas pixels match the face points exactly
function setCamera3D() {
  ortho(-width / 2, width / 2, -height / 2, height / 2, -10000, 10000);
}

function makeShader() {
  const src = `precision mediump float;
varying vec2 vTexCoord;
uniform sampler2D tex0;
uniform vec2 uFace;
uniform float uRadius;
uniform float uFlash;

void main() {
  vec2 uv = vTexCoord;
  vec4 col = texture2D(tex0, uv);
  float d = distance(uv, uFace);
  float light = smoothstep(uRadius * 1.8, uRadius * 0.5, d);
  col.rgb *= 0.45 + 0.65 * light;
  col.rgb += vec3(0.10, 0.04, 0.0) * light;
  col.rgb = mix(col.rgb, vec3(1.0, 0.97, 0.9), uFlash);
  gl_FragColor = vec4(col.rgb, 1.0);
}`;
  shaderError = glslError(src);
  stage = shaderError ? null : createFilterShader(src);
  if (shaderError) setStatus("⚠️ GLSL: " + shaderError);
}

function startExperience() {
  userStartAudio();
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  // 所有导入的声音都经过一个混响，像在戏院里 / all imported sounds go through a reverb, like in a theatre
  reverb = new p5.Reverb();
  reverb.set(2.5, 2);
  reverb.drywet(0.3);
  for (const s of [gong, cymbal, woodblock]) if (s) reverb.process(s, 2.5, 2);
  showCamera = true;   // 你自己在面具后面 / you, behind the mask
  select("#intro").hide();
  started = true;
  loadFaceLandmarker();
}

function draw() {
  translate(-width / 2, -height / 2);   // 让 (0, 0) 在左上角，和 2D 画布一样 / put (0, 0) at the top left, like a 2D canvas
  drawStage();
  if (started) {
    updateFaces();
    faces.forEach((face, i) => {
      const m = MASKS[(maskIndex + i) % MASKS.length];
      stageLights(faceFrame(face));
      if (m.shape) drawMask(face, m);
      else drawOutline(face);
      noLights();
      react(face, i);
    });
    if (showMesh) {
      push();
      translate(0, 0, 3000);   // 画在最前面 / in front of everything
      drawFaceMesh();
      pop();
    }
  }
  drawMaskCard();
  // 舞台灯光（着色器）/ stage light (shader)
  if (stage) {
    const f = faces[0];
    stage.setUniform("uFace", f ? [f.points[FACE.NOSE_TIP].x / width, f.points[FACE.NOSE_TIP].y / height] : [0.5, 0.45]);
    stage.setUniform("uRadius", f ? constrain((faceSize(f) / width) * 1.3, 0.15, 0.6) : 0.35);
    stage.setUniform("uFlash", flash);
    filter(stage);
  }
  flash *= 0.85;
  if (calibrate) showCalibration();
}

// ★★★★★★ 第 1 步 开始 / STEP 1 START：loadMedia（把整个函数换成 Qwen 写的版本 / replace the whole function with Qwen's version）★★★★★★
function loadMedia() {
  // 导入作品需要的所有文件：每一张脸谱的 3D 模型（.obj）和贴图，以及锣、钹、梆子三个声音
  // 导入需要时间，所以都用带回调函数的写法：文件到了以后才存进变量；不要用 preload()
  for (const m of MASKS) {
    // a. 导入 3D 模型；false 表示保持它原来的大小（作品之后会量它）
    loadModel(m.model, false, (shape) => { m.shape = shape; }, () => mediaError(m.model));
    // b. 如果有贴图，就导入贴图
    if (m.texture) {
      loadImage(m.texture, (img) => { m.image = img; }, () => mediaError(m.texture));
    }
  }
  // 锣：张嘴的时候响
  loadSound("media/gong.mp3", (s) => { gong = s; }, () => mediaError("media/gong.mp3"));
  // 钹：变脸的时候响
  loadSound("media/cymbal.mp3", (s) => { cymbal = s; }, () => mediaError("media/cymbal.mp3"));
  // 梆子：微笑的时候敲出五声音阶的节奏
  loadSound("media/woodblock.mp3", (s) => { woodblock = s; }, () => mediaError("media/woodblock.mp3"));
}
// ★★★★★★ 第 1 步 结束 / STEP 1 END ★★★★★★

// ★★★★★★ 第 2 步 开始 / STEP 2 START：drawMask（把整个函数换成 Qwen 写的版本 / replace the whole function with Qwen's version）★★★★★★
function drawMask(face, m) {
  // 把导入的 3D 面具戴到脸上：移到脸的位置，转到脸朝的方向，放大到脸的大小，再贴上它的贴图
  let f = faceFrame(face);
  let box = modelBox(m.shape);
  let k = (f.s * m.size) / box.w;   // 缩放到和脸一样宽
  push();
  // 1. 移到脸的中间
  translate(f.x, f.y, f.z);
  // 2. 转到和脸一样的朝向（矩阵就是脸的三个方向向量 X、Y、Z）
  applyMatrix(f.X.x, f.X.y, f.X.z, 0, f.Y.x, f.Y.y, f.Y.z, 0, f.Z.x, f.Z.y, f.Z.z, 0, 0, 0, 0, 1);
  // 3. 这张面具自己的位置修正
  translate(m.x * f.s, m.y * f.s, m.z * f.s);
  // 4. 这张面具自己的旋转修正
  rotateX(radians(m.turnX));
  rotateY(radians(m.turnY));
  rotateZ(radians(m.turnZ));
  // 5. 放大到和脸一样宽；flipY 把“Y 轴朝上”的模型翻正（p5 里 Y 朝下）
  scale(k, m.flipY ? -k : k, k);
  // 6. 把模型的中心放到原点，这样它绕着自己的中心转
  translate(-box.cx, -box.cy, -box.cz);
  // 7. 贴图：有贴图就贴上去，没有就用灰色
  noStroke();
  if (m.image) {
    texture(m.image);
  } else {
    fill(200);
  }
  // 8. 画出 3D 模型
  model(m.shape);
  pop();
}
// ★★★★★★ 第 2 步 结束 / STEP 2 END ★★★★★★

// ★★★★★★ 第 3 步 开始 / STEP 3 START：faceFrame（把整个函数换成 Qwen 写的版本 / replace the whole function with Qwen's version）★★★★★★
function faceFrame(face) {
  // 量出脸在哪里、有多大，以及在 3D 空间里朝着哪个方向
  let p = face.points;
  let top = p[FACE.FOREHEAD];     // 额头顶部
  let chin = p[FACE.CHIN];        // 下巴
  let left = p[FACE.LEFT_CHEEK];  // 脸的左边缘
  let right = p[FACE.RIGHT_CHEEK]; // 脸的右边缘
  // X：从左边脸颊到右边脸颊的箭头，指向脸的右边（长度变成 1）
  let X = createVector(right.x - left.x, right.y - left.y, right.z - left.z).normalize();
  // Y：从额头到下巴的箭头，先去掉了斜向 X 的部分，让它和 X 正好垂直
  let Y = createVector(chin.x - top.x, chin.y - top.y, chin.z - top.z);
  Y.sub(p5.Vector.mult(X, Y.dot(X)));
  Y.normalize();
  // Z：和 X、Y 都垂直，从脸里朝外，指向摄像头
  let Z = X.cross(Y);
  return {
    x: (top.x + chin.x) / 2,
    y: (top.y + chin.y) / 2,
    z: (top.z + chin.z) / 2,
    s: dist(left.x, left.y, left.z, right.x, right.y, right.z), // 用 3D 量出来的脸宽
    X: X, Y: Y, Z: Z,
  };
}
// ★★★★★★ 第 3 步 结束 / STEP 3 END ★★★★★★

// ★★★★★★ 第 4 步 开始 / STEP 4 START：stageLights（把整个函数换成 Qwen 写的版本 / replace the whole function with Qwen's version）★★★★★★
function stageLights(f) {
  // 像舞台一样给 3D 面具打光，让它显出鼻子、脸颊和额头的立体形状
  // 1. 来自四面八方的柔和暖光：没有地方是全黑的
  ambientLight(110, 90, 80);
  // 2. 主灯：像太阳一样从左上方照下来的平行光
  directionalLight(255, 235, 210, 0.4, 0.5, -1);
  // 3. 侧灯：一盏跟着脸走的橙红色灯，放在脸的旁边
  pointLight(255, 90, 40, f.x + f.s * 1.2, f.y, f.z + f.s);
}
// ★★★★★★ 第 4 步 结束 / STEP 4 END ★★★★★★

// ★★★★★★ 第 5 步 开始 / STEP 5 START：playSounds（把整个函数换成 Qwen 写的版本 / replace the whole function with Qwen's version）★★★★★★
function playSounds(e, prev, changed) {
  // 播放导入的声音：变脸 = 钹，张嘴 = 锣，微笑 = 梆子的五声音阶节奏
  // 声音可能还在导入，所以每次都先检查；只在表情刚出现的那一刻响，而不是每秒 60 次
  // 1. 刚刚变脸：钹
  if (changed && cymbal) {
    cymbal.play();
  }
  // 2. 刚刚张开嘴：锣
  if (e.mouthOpen && !prev.mouthOpen && gong) {
    gong.play();
  }
  // 3. 微笑的时候：用不同的速度敲梆子，音高会变，敲出五声音阶的音
  if (e.smile && frameCount % 12 === 0 && woodblock) {
    woodblock.play(0, random([1, 1.125, 1.25, 1.5, 1.667]), 0.8);
  }
}
// ★★★★★★ 第 5 步 结束 / STEP 5 END ★★★★★★

// ---------- 下面这些已经写好了，不需要改 / already written, no need to change ----------

// 模型有多大、中心在哪里（模型文件自己的单位）/ how big the model is and where its centre is (in the file's own units)
// 第一次用的时候也算出法线（没有法线的模型，打上灯光会是黑的）/ also computes normals the first time (without normals a lit model looks black)
function modelBox(shape) {
  if (!shape.box) {
    if (!shape.vertexNormals.some((n) => n.x !== 0 || n.y !== 0 || n.z !== 0)) shape.computeNormals();   // 灯光需要法线 / lights need normals
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const v of shape.vertices) {
      lo = [min(lo[0], v.x), min(lo[1], v.y), min(lo[2], v.z)];
      hi = [max(hi[0], v.x), max(hi[1], v.y), max(hi[2], v.z)];
    }
    shape.box = { cx: (lo[0] + hi[0]) / 2, cy: (lo[1] + hi[1]) / 2, cz: (lo[2] + hi[2]) / 2, w: max(hi[0] - lo[0], 0.0001) };
  }
  return shape.box;
}

// 读表情 / read the expressions
function readExpression(face) {
  const b = face.blend;
  return {
    mouthOpen: (b.jawOpen || 0) > 0.35,
    browsUp: (b.browInnerUp || 0) > 0.5,
    smile: ((b.mouthSmileLeft || 0) + (b.mouthSmileRight || 0)) / 2 > 0.5,
  };
}

// 表情刚刚出现的那一刻：扬眉 = 变脸 / the moment an expression starts: brows up = change the mask
function react(face, i) {
  const e = readExpression(face);
  const prev = lastExpr[i] || { mouthOpen: false, browsUp: false, smile: false };
  let changed = false;
  if (e.browsUp && !prev.browsUp && frameCount - lastChange > 30) {
    nextMask();
    changed = true;
  }
  playSounds(e, prev, changed);
  lastExpr[i] = e;
}

function nextMask() {
  maskIndex = (maskIndex + 1) % MASKS.length;
  lastChange = frameCount;
  flash = 1;
  showMaskName();
}

function showMaskName() {
  select("#mask-name").html(MASKS[maskIndex].name);
}

// 舞台：红色幕布，后面是你自己（摄像头），你的眼睛会从面具的眼洞里看出来
// the stage: a red curtain, and you (the camera) behind it, so your eyes look out through the mask's eye holes
function drawStage() {
  background(70, 8, 14);
  push();
  translate(0, 0, -3000);
  noStroke();
  for (let x = 0; x < width; x += 24) {
    fill(110, 14, 22, 60 + 50 * sin(x * 0.05));
    rect(x, 0, 12, height);
  }
  if (started && showCamera && video && video.elt.readyState >= 2) {
    translate(0, 0, 10);
    const fit = videoFit();
    push();
    translate(width, 0);
    scale(-1, 1);
    tint(255, 200, 170, 230);
    image(video, fit.x, fit.y, fit.w, fit.h);
    pop();
  }
  pop();
}

// 左上角的小卡片：现在这张脸谱的贴图（摊平的样子）/ a small card in the top left: the current mask's texture (flat)
function drawMaskCard() {
  const img = MASKS[maskIndex].image;
  if (!img) return;
  push();
  translate(0, 0, 3000);
  noStroke();
  fill(255, 236, 200);
  rect(14, 52, 104, 104, 6);
  image(img, 16, 54, 100, 100);
  pop();
}

// 模型还没有导入时：画出脸的轮廓 / before the model is imported: draw the outline of the face
function drawOutline(face) {
  push();
  translate(0, 0, 20);
  noFill();
  stroke(255, 200);
  strokeWeight(2);
  beginShape();
  for (const i of FACE_OVAL) vertex(face.points[i].x, face.points[i].y);
  endShape(CLOSE);
  pop();
}

// ---------- 校准模式：给新的面具找到合适的 size、x、y、turnX…… / calibration: find size, x, y, turnX… for a new mask ----------
// C：打开/关闭 · ←→↑↓：移动 · + −：大小 · W S：前后 · A D：左右转 · Q E：上下转 · Z X：歪头 · F：翻转 · P：打印设置
// C: on/off · arrows: move · + −: size · W S: forward/back · A D: turn · Q E: tilt · Z X: roll · F: flip · P: print
function calibrateKey() {
  const m = MASKS[maskIndex];
  const step = 0.02, turn = 5;
  if (keyCode === LEFT_ARROW) m.x -= step;
  if (keyCode === RIGHT_ARROW) m.x += step;
  if (keyCode === UP_ARROW) m.y -= step;
  if (keyCode === DOWN_ARROW) m.y += step;
  if (key === "+" || key === "=") m.size += 0.05;
  if (key === "-" || key === "_") m.size = max(0.05, m.size - 0.05);
  if (key === "w" || key === "W") m.z += step;
  if (key === "s" || key === "S") m.z -= step;
  if (key === "a" || key === "A") m.turnY -= turn;
  if (key === "d" || key === "D") m.turnY += turn;
  if (key === "q" || key === "Q") m.turnX -= turn;
  if (key === "e" || key === "E") m.turnX += turn;
  if (key === "z" || key === "Z") m.turnZ -= turn;
  if (key === "x" || key === "X") m.turnZ += turn;
  if (key === "f" || key === "F") m.flipY = !m.flipY;
  if (key === "p" || key === "P") console.log(maskSettings(m));
}

function maskSettings(m) {
  const r = (v) => round(v * 100) / 100;
  return `size: ${r(m.size)}, x: ${r(m.x)}, y: ${r(m.y)}, z: ${r(m.z)}, turnX: ${m.turnX}, turnY: ${m.turnY}, turnZ: ${m.turnZ}, flipY: ${m.flipY}`;
}

function showCalibration() {
  setStatus("🔧 校准 / calibrate — " + maskSettings(MASKS[maskIndex]) + " — P = 打印 / print");
}

function keyPressed() {
  if (key === "c" || key === "C") {
    calibrate = !calibrate;
    if (!calibrate) setStatus("");
    return false;
  }
  if (calibrate) {
    calibrateKey();
    return false;
  }
  if (key === " ") nextMask();                        // 空格也能变脸 / space also changes the mask
  if (key === "g" || key === "G") { if (gong) gong.play(); }
  if (key === "s" || key === "S") saveCanvas("my-opera-mask", "png");
  if (key === "v" || key === "V") showCamera = !showCamera;
  if (key === "k" || key === "K") showMesh = !showMesh;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  setCamera3D();
}
