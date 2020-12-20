import {mat4} from 'gl-matrix';

/**
 * Weights functions
 */

const layer_1_depth = 16; // Convolution depth
const layer_1_width = 5; // Convolution width

const layer_2_depth = 8;
const layer_2_width = 3;

const reconstruct_width = 3;

function get_conv1_1_weights(raw_weights) {
  const conv1_1_weights = [];

  for (let y = 0; y < layer_1_width; y++) {
    for (let outw = 0; outw < layer_1_depth; outw++) {
      for (let inw = 0; inw < 3; inw++) {
        conv1_1_weights.push(raw_weights["W_conv1_1"][y][0][inw][outw]);
      }
    }
  }

  return conv1_1_weights;
}

function get_conv1_2_weights(raw_weights) {
  const conv1_2_weights = [];

  for (let x = 0; x < layer_1_width; x++) {
    for (let outw = 0; outw < layer_1_depth; outw++) {
      for (let inw = 0; inw < layer_1_depth; inw++) {
        conv1_2_weights.push(raw_weights["W_conv1_2"][0][x][inw][outw]);
      }
    }
  }

  return conv1_2_weights;
}

function get_conv1_biases(raw_weights) {
  return raw_weights["b_conv1"];
}

function get_conv2_1_weights(raw_weights) {
  const conv2_1_weights = [];

  for (let y = 0; y < layer_2_width; y++) {
    for (let outw = 0; outw < layer_2_depth; outw++) {
      for (let inw = 0; inw < layer_1_depth; inw++) {
        conv2_1_weights.push(raw_weights["W_conv2_1"][y][0][inw][outw]);
      }
    }
  }

  return conv2_1_weights;
}

function get_conv2_2_weights(raw_weights) {
  const conv2_2_weights = [];

  for (let x = 0; x < layer_2_width; x++) {
    for (let outw = 0; outw < layer_2_depth; outw++) {
      for (let inw = 0; inw < layer_2_depth; inw++) {
        conv2_2_weights.push(raw_weights["W_conv2_2"][0][x][inw][outw]);
      }
    }
  }

  return conv2_2_weights;
}

function get_conv2_biases(raw_weights) {
  return raw_weights["b_conv2"];
}

/**
 * XXX: These weights organized differently to enable vec4 dot product
 */
function get_reconstruct_weights(raw_weights) {
  const conv_reconstruct_weights = [];

  for (let out_y = 0; out_y < 3; out_y++) {
    for (let out_x = 0; out_x < 3; out_x++) {
      for (let j = 0; j < reconstruct_width; j++) {
        for (let i = 0; i < reconstruct_width; i++) {
          for (let out_pixel = 0; out_pixel < 3; out_pixel++) {
            for (let z = 0; z < layer_2_depth; z++) {
              conv_reconstruct_weights.push(raw_weights["W_reconstruct"][j][i][z][out_y * 9 + out_x * 3 + out_pixel]);
            }
          }
        }
      }
    }
  }

  return conv_reconstruct_weights;
}

function get_reconstruct_biases(raw_weights) {
  return raw_weights["b_reconstruct"];
}

/**
 * Utility functions
 */
const vsSource = `#version 300 es
  #pragma vscode_glsllint_stage: vert

  in vec4 aVertexPosition;
  uniform mat4 uProjectionModelViewMatrix;

  void main(void) {
      gl_Position = uProjectionModelViewMatrix * aVertexPosition;
  }
  `;

// creates a shader of the given type, uploads the source and compiles it.
function loadShader(gl, type, source) {
  const shader = gl.createShader(type);

  gl.shaderSource(shader, source);

  gl.compileShader(shader);

  // See if it compiled successfully
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Initialize a shader program, so WebGL knows how to draw our data
function initShaderProgram(gl, vsSource, fsSource) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

  const shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);

  // If linking the shader program failed, alert
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
    return null;
  }

  return shaderProgram;
}

/**
 * WebGL Shaders
 */
// Symmetrically pad a 2d texture with black
function initPadProgram(gl, padding) {
  const padFragShader = `#version 300 es
  #pragma vscode_glsllint_stage: frag

  precision mediump float;

  uniform sampler2D originalSampler;
  uniform vec2 videoRes;

  vec2 padding = vec2(${padding});
  layout(location = 0) out vec4 padOut;

  void main() {
    // mask.xy == (1.0, 1.0) when inside the padding
    vec2 mask = step(padding.xy, gl_FragCoord.xy) - step(videoRes.xy + padding, gl_FragCoord.xy);

    // adjust texture coords to account for padding
    vec2 coords = (padding * vec2(-1.0) + gl_FragCoord.xy) / videoRes.xy;    

    padOut = mask.x * mask.y * texture(originalSampler, coords) * 255.0;
  }
  `;

  console.log(padFragShader);

  return initShaderProgram(gl, vsSource, padFragShader);
}


// Vertical conv2d
// in: width x height x 3
// out: width x (height-4) x 16
// kernel size 1x5
function init_conv1_1_program(gl) {
  const operations = [];

  for (let i = 0; i < 5; i++) {
    operations.push(`
      coords = vec2(gl_FragCoord.x * videoResInverse.x, (gl_FragCoord.y + ${i}.0) * videoResInverse.y);

      texData = texture(padSampler, coords).rgb;
      

      out0 += vec4(dot(texData, weights[${i * layer_1_depth + 0}]),
                        dot(texData, weights[${i * layer_1_depth + 1}]),
                        dot(texData, weights[${i * layer_1_depth + 2}]),
                        dot(texData, weights[${i * layer_1_depth + 3}]));
      
      out1 += vec4(dot(texData, weights[${i * layer_1_depth + 4}]),
                        dot(texData, weights[${i * layer_1_depth + 5}]),
                        dot(texData, weights[${i * layer_1_depth + 6}]),
                        dot(texData, weights[${i * layer_1_depth + 7}]));
      
      out2 += vec4(dot(texData, weights[${i * layer_1_depth + 8}]),
                        dot(texData, weights[${i * layer_1_depth + 9}]),
                        dot(texData, weights[${i * layer_1_depth + 10}]),
                        dot(texData, weights[${i * layer_1_depth + 11}]));
      
      out3 += vec4(dot(texData, weights[${i * layer_1_depth + 12}]),
                        dot(texData, weights[${i * layer_1_depth + 13}]),
                        dot(texData, weights[${i * layer_1_depth + 14}]),
                        dot(texData, weights[${i * layer_1_depth + 15}]));
    `);
  }

  const conv1_1_shader = `#version 300 es
  #pragma vscode_glsllint_stage: frag

  precision mediump float;

  uniform sampler2D padSampler;
  uniform vec3 weights[${layer_1_width * layer_1_depth}];
  uniform vec2 videoRes;

  layout(location = 0) out vec4 out0;
  layout(location = 1) out vec4 out1;
  layout(location = 2) out vec4 out2;
  layout(location = 3) out vec4 out3;

  void main() {
    vec2 coords = vec2(0.0);
    vec3 texData = vec3(0.0);

    out0 = vec4(0.0);
    out1 = vec4(0.0);
    out2 = vec4(0.0);
    out3 = vec4(0.0);

    vec2 videoResInverse = 1.0 / (videoRes + 8.0);

    // Operations
    ${operations.join("\n")}
  }
  `;

  console.log(conv1_1_shader)

  return initShaderProgram(gl, vsSource, conv1_1_shader);
}


// Vertical conv2d
// in: width x height x 8
// out: (width-4) x height x 8
// kernel size 5x1
function init_conv1_2_program(gl) {
  const operations = [];

  for (let i = 0; i < layer_1_width; i++) {
    operations.push(`
      coords = vec2((gl_FragCoord.x + ${i}.0) * inWidthInverse, gl_FragCoord.y * inHeightInverse);

      in_0 = texture(layer1Sampler, coords);
      in_1 = texture(layer2Sampler, coords);
      in_2 = texture(layer3Sampler, coords);
      in_3 = texture(layer4Sampler, coords);

      out0.rgba += vec4(dot(in_0, weights[${i * layer_1_depth * 4 + 0}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 1}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 2}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 3}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 4}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 5}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 6}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 7}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 8}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 9}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 10}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 11}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 12}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 13}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 14}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 15}]));
      
      out1.rgba += vec4(dot(in_0, weights[${i * layer_1_depth * 4 + 16}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 17}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 18}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 19}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 20}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 21}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 22}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 23}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 24}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 25}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 26}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 27}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 28}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 29}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 30}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 31}]));
      
      out2.rgba += vec4(dot(in_0, weights[${i * layer_1_depth * 4 + 32}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 33}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 34}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 35}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 36}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 37}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 38}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 39}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 40}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 41}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 42}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 43}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 44}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 45}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 46}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 47}]));
                        
      out3.rgba += vec4(dot(in_0, weights[${i * layer_1_depth * 4 + 48}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 49}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 50}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 51}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 52}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 53}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 54}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 55}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 56}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 57}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 58}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 59}]),
                        dot(in_0, weights[${i * layer_1_depth * 4 + 60}]) + dot(in_1, weights[${i * layer_1_depth * 4 + 61}]) + dot(in_2, weights[${i * layer_1_depth * 4 + 62}]) + dot(in_3, weights[${i * layer_1_depth * 4 + 63}]));
    `);
  }

  const conv1_2_shader = `#version 300 es
  #pragma vscode_glsllint_stage: frag

  precision mediump float;
  precision mediump sampler2D;

  uniform sampler2D layer1Sampler;
  uniform sampler2D layer2Sampler;
  uniform sampler2D layer3Sampler;
  uniform sampler2D layer4Sampler;

  uniform vec4 weights[${layer_1_width * layer_1_depth * 4}];
  uniform vec4 biases[4];
  uniform vec2 videoRes;

  layout(location = 0) out vec4 out0;
  layout(location = 1) out vec4 out1;
  layout(location = 2) out vec4 out2;
  layout(location = 3) out vec4 out3;

  void main() {
    vec2 coords = vec2(0.0);
    vec4 in_0 = vec4(0.0);
    vec4 in_1 = vec4(0.0);
    vec4 in_2 = vec4(0.0);
    vec4 in_3 = vec4(0.0);

    out0 = vec4(0.0);
    out1 = vec4(0.0);
    out2 = vec4(0.0);
    out3 = vec4(0.0);

    float inWidthInverse = 1.0 / (videoRes.x + 8.0);
    float inHeightInverse = 1.0 / (videoRes.y + 4.0);

    // Operations
    ${operations.join("\n")}

    out0 = max(out0 + biases[0], 0.0);
    out1 = max(out1 + biases[1], 0.0);
    out2 = max(out2 + biases[2], 0.0);
    out3 = max(out3 + biases[3], 0.0);
  }
  `;

  console.log(conv1_2_shader)

  return initShaderProgram(gl, vsSource, conv1_2_shader);
}


// Vertical conv2d
// in: width x height x 16
// out: width x (height - 2) x 8
// kernel size 1 x 3
function init_conv2_1_program(gl) {
  const operations = [];

  for (let i = 0; i < layer_2_width; i++) {
    operations.push(`
      coords = vec2(gl_FragCoord.x * videoResInverse.x, (gl_FragCoord.y + ${i}.0) * videoResInverse.y);

      in_0 = texture(layer1Sampler, coords);
      in_1 = texture(layer2Sampler, coords);
      in_2 = texture(layer3Sampler, coords);
      in_3 = texture(layer4Sampler, coords);

      out0.rgba += vec4(dot(in_0, weights[${i * layer_2_depth * 4 + 0}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 1}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 2}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 3}]),
                        dot(in_0, weights[${i * layer_2_depth * 4 + 4}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 5}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 6}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 7}]),
                        dot(in_0, weights[${i * layer_2_depth * 4 + 8}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 9}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 10}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 11}]),
                        dot(in_0, weights[${i * layer_2_depth * 4 + 12}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 13}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 14}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 15}]));
      
      out1.rgba += vec4(dot(in_0, weights[${i * layer_2_depth * 4 + 16}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 17}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 18}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 19}]),
                        dot(in_0, weights[${i * layer_2_depth * 4 + 20}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 21}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 22}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 23}]),
                        dot(in_0, weights[${i * layer_2_depth * 4 + 24}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 25}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 26}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 27}]),
                        dot(in_0, weights[${i * layer_2_depth * 4 + 28}]) + dot(in_1, weights[${i * layer_2_depth * 4 + 29}]) + dot(in_2, weights[${i * layer_2_depth * 4 + 30}]) + dot(in_3, weights[${i * layer_2_depth * 4 + 31}]));
    `);
  }

  const conv2_1_shader = `#version 300 es
  #pragma vscode_glsllint_stage: frag

  precision mediump float;

  uniform sampler2D layer1Sampler;
  uniform sampler2D layer2Sampler;
  uniform sampler2D layer3Sampler;
  uniform sampler2D layer4Sampler;
  uniform vec2 videoRes;

  uniform vec4 weights[${layer_2_width * layer_2_depth * 4}];

  layout(location = 0) out vec4 out0;
  layout(location = 1) out vec4 out1;

  void main() {
    vec2 coords = vec2(0.0);
    vec4 in_0 = vec4(0.0);
    vec4 in_1 = vec4(0.0);
    vec4 in_2 = vec4(0.0);
    vec4 in_3 = vec4(0.0);
  
    out0 = vec4(0.0);
    out1 = vec4(0.0);

    vec2 videoResInverse = 1.0 / (videoRes + 4.0);

    // Operations
    ${operations.join("\n")}
  }
  `;

  console.log(conv2_1_shader)

  return initShaderProgram(gl, vsSource, conv2_1_shader);
}

// Vertical conv2d
// in: width x height x 4
// out: (width-2) x height x 4
// kernel size 3 x 1
function init_conv2_2_program(gl) {
  const operations = [];

  for (let i = 0; i < 3; i++) {
    operations.push(`
      coords = vec2((gl_FragCoord.x + ${i}.0) * inWidthInverse, gl_FragCoord.y * inHeightInverse);
      
      in_0 = texture(layer1Sampler, coords);
      in_1 = texture(layer2Sampler, coords);
      
      out0.rgba += vec4(dot(in_0, weights[${i * 8 * 2 + 0}]) + dot(in_1, weights[${i * 8 * 2 + 1}]),
                        dot(in_0, weights[${i * 8 * 2 + 2}]) + dot(in_1, weights[${i * 8 * 2 + 3}]),
                        dot(in_0, weights[${i * 8 * 2 + 4}]) + dot(in_1, weights[${i * 8 * 2 + 5}]),
                        dot(in_0, weights[${i * 8 * 2 + 6}]) + dot(in_1, weights[${i * 8 * 2 + 7}]));
      
      out1.rgba += vec4(dot(in_0, weights[${i * 8 * 2 + 8}]) + dot(in_1, weights[${i * 8 * 2 + 9}]),
                        dot(in_0, weights[${i * 8 * 2 + 10}]) + dot(in_1, weights[${i * 8 * 2 + 11}]),
                        dot(in_0, weights[${i * 8 * 2 + 12}]) + dot(in_1, weights[${i * 8 * 2 + 13}]),
                        dot(in_0, weights[${i * 8 * 2 + 14}]) + dot(in_1, weights[${i * 8 * 2 + 15}]));
    `);
  }

  const conv2_2_shader = `#version 300 es
  #pragma vscode_glsllint_stage: frag
  
  precision mediump float;

  uniform sampler2D layer1Sampler;
  uniform sampler2D layer2Sampler;

  uniform vec4 weights[${layer_2_width * layer_2_depth * 2}];
  uniform vec4 biases[2];
  uniform vec2 videoRes;

  layout(location = 0) out vec4 out0;
  layout(location = 1) out vec4 out1;

  void main() {    
    vec2 coords = vec2(0.0);
    vec4 in_0 = vec4(0.0);
    vec4 in_1 = vec4(0.0);  
  
    out0 = vec4(0.0);
    out1 = vec4(0.0);
    
    // divisions are more expensive than multiplications, calculate it once and use it as a multiplication many times 
    float inWidthInverse = 1.0 / (videoRes.x + 4.0);
    float inHeightInverse = 1.0 / (videoRes.y + 2.0);

    // Operations
    ${operations.join("\n")}

    out0 = max(out0 + biases[0], 0.0);
    out1 = max(out1 + biases[1], 0.0);
  }
  `;

  console.log(conv2_2_shader)

  return initShaderProgram(gl, vsSource, conv2_2_shader);
}

// sub-pixel convolutional layer
// in width x height x 4
// out (width - 2) * 3 x (height - 2) * 3 x 3
// kernel 3 x 3
function init_reconstruct_program(gl) {
  const coords = [];
  const inputs = [];
  const weights = [];
  const operations = [];

  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 3; i++) {
      // Todo
      coords.push(`vec2 coords_${j}_${i} = vec2((fIn.x + ${i}.0) * videoResInverse.x, (fIn.y + ${j}.0) * videoResInverse.y);`);

      inputs.push(`vec4 in_${j}_${i}_0 = texture(layer1Sampler, coords_${j}_${i});`);
      inputs.push(`vec4 in_${j}_${i}_1 = texture(layer2Sampler, coords_${j}_${i});`);

      weights.push(`vec4 w_${j}_${i}_0_0 = weights[2 * (iOut.y * 81 + iOut.x * 27 + ${j * 9 + i * 3}) + 0];`);
      weights.push(`vec4 w_${j}_${i}_0_1 = weights[2 * (iOut.y * 81 + iOut.x * 27 + ${j * 9 + i * 3}) + 1];`);
      weights.push(`vec4 w_${j}_${i}_1_0 = weights[2 * (iOut.y * 81 + iOut.x * 27 + ${j * 9 + i * 3}) + 2];`);
      weights.push(`vec4 w_${j}_${i}_1_1 = weights[2 * (iOut.y * 81 + iOut.x * 27 + ${j * 9 + i * 3}) + 3];`);
      weights.push(`vec4 w_${j}_${i}_2_0 = weights[2 * (iOut.y * 81 + iOut.x * 27 + ${j * 9 + i * 3}) + 4];`);
      weights.push(`vec4 w_${j}_${i}_2_1 = weights[2 * (iOut.y * 81 + iOut.x * 27 + ${j * 9 + i * 3}) + 5];`);

      operations.push(`out0.rgb += vec3(dot(in_${j}_${i}_0, w_${j}_${i}_0_0) + dot(in_${j}_${i}_1, w_${j}_${i}_0_1),
                                        dot(in_${j}_${i}_0, w_${j}_${i}_1_0) + dot(in_${j}_${i}_1, w_${j}_${i}_1_1),
                                        dot(in_${j}_${i}_0, w_${j}_${i}_2_0) + dot(in_${j}_${i}_1, w_${j}_${i}_2_1));`);
    }
  }

  const reconstruct_shader = `#version 300 es
  #pragma vscode_glsllint_stage: frag
  
  precision mediump float;

  uniform sampler2D originalSampler;
  uniform sampler2D layer1Sampler;
  uniform sampler2D layer2Sampler;
  uniform vec2 videoRes;

  uniform sampler2D maskSampler;

  uniform vec4 weights[${3 * 3 * 3 * 9 * 2}];
  uniform vec3 biases[9];

  const float oneThird = 1.0 / 3.0;
  const float oneTwoFiftyFifth = 1.0 / 255.0;
  
  out vec4 out0;

  void main() {
    out0 = vec4(0.0, 0.0, 0.0, 1.0);

    ivec2 iOut = ivec2(mod(gl_FragCoord - 0.5, 3.0));
    vec2 fIn = vec2(gl_FragCoord - float(iOut) + 1.0) * oneThird;
    vec2 videoResInverse = 1.0 / (videoRes + 2.0);

    // Coords
${coords.join("\n")}

    // Inputs
${inputs.join("\n")}

    // Weights
${weights.join("\n")}

    // Operations
${operations.join("\n")}

    out0.rgb = (out0.rgb + biases[3 * iOut.y + iOut.x].rgb) * oneTwoFiftyFifth;
    out0.rgb += texture(originalSampler, (gl_FragCoord.xy / (videoRes * 3.0))).rgb;
    out0.rgb = clamp(out0.rgb, 0.0, 1.0);
  }
  `;

  console.log(reconstruct_shader);

  return initShaderProgram(gl, vsSource, reconstruct_shader);
}

function initRenderProgram(gl) {
  const renderFragShader = `#version 300 es
  #pragma vscode_glsllint_stage: frag

  precision mediump float;

  uniform sampler2D originalSampler;

  uniform vec2 videoRes;
  uniform vec4 renderArea;

  layout(location = 0) out vec4 copyOut;

  void main() {
    // check if the gl_FragCoord is within the bounds of the renderArea
    // if mask.x == 1.0 it means we are within the x bounds of renderArea, similarly for mask.y
    vec2 mask = step(renderArea.xy, gl_FragCoord.xy) - step(renderArea.zw, gl_FragCoord.xy);

    // align the image in the renderArea area and scale to the videoRes
    vec2 texCoords = (gl_FragCoord.xy - renderArea.xy) / videoRes.xy;
    // flip the texture image vertically
    texCoords.y = 1.0 - texCoords.y;

    // if mask.x and mask.y are 1.0 use the value returned from texture()  
    copyOut = mask.x * mask.y * texture(originalSampler, texCoords);
  }
  `;

  console.log(renderFragShader);

  return initShaderProgram(gl, vsSource, renderFragShader);
}


function initBuffers(gl) {
  // Create a buffer for the cube's vertex positions.
  const positionBuffer = gl.createBuffer();

  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  // Now create an array of positions for the cube.
  const positions = [
    // Simple square
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0
  ];

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  // Build the element array buffer; this specifies the indices
  // into the vertex arrays for each face's vertices.

  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

  const indices = [
    2,
    0,
    3,
    1,
    0,
    2 // front
  ];

  // Now send the element array to GL
  gl.bufferData(
    gl.ELEMENT_ARRAY_BUFFER,
    new Uint16Array(indices),
    gl.STATIC_DRAW
  );

  return {
    position: positionBuffer,
    indices: indexBuffer
  };
}

/**
 * Create a new texture to store the video frame data
 *
 * @param gl
 * @param width
 * @param height
 * @param float
 * @returns {WebGLTexture}
 */
function createTexture(gl, width, height, float) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  {
    const level = 0;
    const border = 0;
    const format = gl.RGBA;
    let internalFormat;
    let type;

    if (float) {
      internalFormat = gl.RGBA16F;
      type = gl.FLOAT;
    } else {
      internalFormat = gl.RGBA;
      type = gl.UNSIGNED_BYTE;
    }
    const data = null;
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      format,
      type,
      data
    );

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  return texture;
}


/**
 * Initialize a texture and load an image. When the image is finished loading
 * copy it into the texture.
 *
 * @param gl
 * @returns {WebGLTexture}
 */
function initTexture(gl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  const level = 0;
  const internalFormat = gl.RGBA;
  const width = 1;
  const height = 1;
  const border = 0;
  const srcFormat = gl.RGBA;
  const srcType = gl.UNSIGNED_BYTE;
  const pixel = new Uint8Array([0, 0, 0, 255]); // opaque blue
  gl.texImage2D(
    gl.TEXTURE_2D,
    level,
    internalFormat,
    width,
    height,
    border,
    srcFormat,
    srcType,
    pixel
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  return texture;
}

/**
 * Set the texture to the current video frame
 *
 * @param gl
 * @param texture
 * @param video
 */
function updateTexture(gl, texture, video) {
  const level = 0;
  const internalFormat = gl.RGBA;
  const srcFormat = gl.RGBA;
  const srcType = gl.UNSIGNED_BYTE;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    level,
    internalFormat,
    srcFormat,
    srcType,
    video
  );
}

// Setup the unchanging variables for drawScene

// the multiplication of projectionMatrix and modelViewMatrix are used in the vertex shader, we compute it here once
// and pass in the result as a uniform

const projectionModelViewMatrix = mat4.create();

{
  // Create a perspective matrix, a special matrix that is
  // used to simulate the distortion of perspective in a camera.
  // Our field of view is 45 degrees, with a width/height
  // ratio that matches the display size of the canvas
  // and we only want to see objects between 0.1 units
  // and 100 units away from the camera.
  const zNear = 0.1;
  const zFar = 100.0;
  const projectionMatrix = mat4.create();
  mat4.ortho(projectionMatrix, -1.0, 1.0, 1.0, -1.0, zNear, zFar);

  // Set the drawing position to the "identity" point, which is
  // the center of the scene.
  const modelViewMatrix = mat4.create();

  // Now move the drawing position a bit to where we want to
  // start drawing the square.
  mat4.translate(
    modelViewMatrix, // destination matrix
    modelViewMatrix, // matrix to translate
    [-0.0, 0.0, -6.0] // amount to translate
  );

  const normalMatrix = mat4.create();
  mat4.invert(normalMatrix, modelViewMatrix);
  mat4.transpose(normalMatrix, normalMatrix);

  mat4.multiply(projectionModelViewMatrix, projectionMatrix, modelViewMatrix);
}

/**
 * Executes the GL program with the passed buffers and texture
 *
 * @param gl
 * @param programInfo
 * @param buffers
 * @param texture
 */
function drawScene(gl, programInfo, buffers, texture) {
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Tell WebGL how to pull out the positions from the position
  // buffer into the vertexPosition attribute
  {
    const numComponents = 3;
    const type = gl.FLOAT;
    const normalize = false;
    const stride = 0;
    const offset = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(
      programInfo.attribLocations.vertexPosition,
      numComponents,
      type,
      normalize,
      stride,
      offset
    );
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
  }

  // Tell WebGL which indices to use to index the vertices
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);

  gl.useProgram(programInfo.program);

  // Set the shader uniforms
  gl.uniformMatrix4fv(
    programInfo.uniformLocations.projectionModelViewMatrix,
    false,
    projectionModelViewMatrix,
  );
  if (programInfo.textures.length > 0) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, programInfo.textures[0]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, programInfo.filters[0]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, programInfo.filters[0]);

    gl.uniform1i(programInfo.samplers[0], 0);
  }

  if (programInfo.textures.length > 1) {
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, programInfo.textures[1]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, programInfo.filters[1]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, programInfo.filters[1]);

    gl.uniform1i(programInfo.samplers[1], 1);
  }

  if (programInfo.textures.length > 2) {
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, programInfo.textures[2]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, programInfo.filters[2]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, programInfo.filters[2]);

    gl.uniform1i(programInfo.samplers[2], 2);
  }

  if (programInfo.textures.length > 3) {
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, programInfo.textures[3]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, programInfo.filters[3]);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, programInfo.filters[3]);

    gl.uniform1i(programInfo.samplers[3], 3);
  }

  if (programInfo.weights) {
    if (programInfo.rgbWeights) {
      gl.uniform3fv(
        programInfo.uniformLocations.weightsLocation,
        programInfo.weights
      );
    } else {
      gl.uniform4fv(
        programInfo.uniformLocations.weightsLocation,
        programInfo.weights
      );
    }
  }

  if (programInfo.biases) {
    if (programInfo.rgbBiases) {
      gl.uniform3fv(
        programInfo.uniformLocations.biasesLocation,
        programInfo.biases
      );
    } else {
      gl.uniform4fv(
        programInfo.uniformLocations.biasesLocation,
        programInfo.biases
      );
    }
  }

  if (programInfo.renderArea) {
    gl.uniform4fv(programInfo.uniformLocations.renderAreaLocation, programInfo.renderArea);
  }
  if (programInfo.videoRes) {
    gl.uniform2fv(programInfo.uniformLocations.videoResLocation, programInfo.videoRes);
  }

  {
    const vertexCount = 6;
    const type = gl.UNSIGNED_SHORT;
    const offset = 0;
    gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
  }
}

function resizeCanvas(canvas) {
  const displayWidth = canvas.clientWidth;
  const displayHeight = canvas.clientHeight;

  // Check if the canvas is not the same size.
  if (canvas.width !== displayWidth ||
    canvas.height !== displayHeight) {

    // Make the canvas the same size
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
}

/**
 * Returns the coordinates to center the video within the canvas
 *
 * @param vw
 * @param vh
 * @param cw
 * @param ch
 * @returns {[number, number, number, number]}
 */
const fitBoundingBox = (vw, vh, cw, ch) => {
  let left = Math.round((cw - vw) * 0.5);
  const top = Math.round((ch - vh) * 0.5);
  return [left, top, cw - left, ch - top];
}

/**
 * Return the size of the video and the render area
 *
 * @param videoWidth
 * @param videoHeight
 * @param canvasWidth
 * @param canvasHeight
 * @returns {(number[]|[number, number, number, number])[]}
 */
function scaleToFit(videoWidth, videoHeight, canvasWidth, canvasHeight) {
  let curWidth = videoWidth;
  let curHeight = videoHeight;

  const scaleFactor = Math.min(canvasWidth / videoWidth, canvasHeight / videoHeight);

  curWidth = Math.ceil(scaleFactor * curWidth);
  curHeight = Math.floor(scaleFactor * curHeight);

  // if video is vertical
  if (videoHeight > videoWidth) {
    return [[curHeight, curWidth], fitBoundingBox(curHeight, curWidth, canvasWidth, canvasHeight)];
  }
  return [[curWidth, curHeight], fitBoundingBox(curWidth, curHeight, canvasWidth, canvasHeight)];
}

/**
 * Start here
 *
 * @param player The VideoJS player
 * @param canvas The HTML canvas
 * @param options Options passed to the player
 */
export function main(player, canvas, options) {
  const video = player.tech().el();
  const gl = canvas.getContext('webgl2');
  let videoHeight = video.videoHeight || 286;
  let videoWidth = video.videoWidth || 640;
  const renderArea = [0, 0, 100, 100];
  const videoRes = [100, 100];
  const targetFrameRate = parseInt(options.frameRate) || 30;
  // set to true when video can be copied to texture, ie. when the video is loaded and playing
  let copyVideo = false;

  player.on('playing', () => {
    copyVideo = true;
    requestAnimationFrame(render);
  });

  player.on(['pause', 'ended'], () => {
    copyVideo = false;
  });

  if (!gl) {
    alert('Unable to initialize WebGL. Your browser or machine may not support it.');
    return;
  }

  // enable WebGL extensions
  gl.getExtension('EXT_color_buffer_float');
  gl.getExtension('OES_texture_float_linear');
  gl.getExtension('OES_texture_half_float_linear');

  // set default clearing values
  gl.clearColor(0.0, 0.0, 0.0, 1.0); // Clear to black, fully opaque
  gl.enable(gl.DEPTH_TEST); // Enable depth testing
  gl.depthFunc(gl.LEQUAL); // Near things obscure far things


  // Initialize the textures

  // Input
  const input_texture = initTexture(gl);

  // Padded input
  let pad_texture = createTexture(gl, videoWidth + 8, videoHeight + 8, true);

  let conv1_1_texture1 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);
  let conv1_1_texture2 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);
  let conv1_1_texture3 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);
  let conv1_1_texture4 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);

  let conv1_2_texture1 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);
  let conv1_2_texture2 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);
  let conv1_2_texture3 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);
  let conv1_2_texture4 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);

  let conv2_1_texture1 = createTexture(gl, videoWidth + 4, videoHeight + 2, true);
  let conv2_1_texture2 = createTexture(gl, videoWidth + 4, videoHeight + 2, true);

  // W_conv2_2: in 644x288x4 out (w+4)x(h+4)x4
  let conv2_2_texture1 = createTexture(gl, videoWidth + 2, videoHeight + 2, true);
  let conv2_2_texture2 = createTexture(gl, videoWidth + 2, videoHeight + 2, true);

  // W_reconstruct: in 642x288x4 out 1920x858x3
  let reconstruct_texture = createTexture(gl, videoWidth * 3, videoHeight * 3, false);

  console.log("pad program");
  const padProgram = initPadProgram(gl, 4);
  const padProgramInfo = {
    program: padProgram,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(padProgram, 'aVertexPosition'),
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(padProgram, 'uProjectionModelViewMatrix'),
      videoResLocation: gl.getUniformLocation(padProgram, 'videoRes')
    },
    // 1-1 mapping between samplers and textures
    samplers: [gl.getUniformLocation(padProgram, 'originalSampler')],
    textures: [input_texture],
    filters: [gl.NEAREST],
    videoRes: videoRes
  };

  console.log("conv1_1 program");
  const conv1_1_program = init_conv1_1_program(gl);
  const conv1_1_program_info = {
    program: conv1_1_program,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(conv1_1_program, 'aVertexPosition'),
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(conv1_1_program, 'uProjectionModelViewMatrix'),
      weightsLocation: gl.getUniformLocation(conv1_1_program, 'weights'),
      videoResLocation: gl.getUniformLocation(conv1_1_program, 'videoRes')
    },
    samplers: [gl.getUniformLocation(conv1_1_program, 'padSampler')],
    textures: [pad_texture],
    filters: [gl.NEAREST],
    weights: get_conv1_1_weights(options.weights),
    rgbWeights: true,
    videoRes: videoRes
  };

  console.log("conv1_2 program");
  const conv1_2_program = init_conv1_2_program(gl);
  const conv1_2_program_info = {
    program: conv1_2_program,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(conv1_2_program, 'aVertexPosition'),
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(conv1_2_program, 'uProjectionModelViewMatrix'),
      weightsLocation: gl.getUniformLocation(conv1_2_program, 'weights'),
      biasesLocation: gl.getUniformLocation(conv1_2_program, 'biases'),
      videoResLocation: gl.getUniformLocation(conv1_2_program, 'videoRes')
    },
    samplers: [
      gl.getUniformLocation(conv1_2_program, 'layer1Sampler'),
      gl.getUniformLocation(conv1_2_program, 'layer2Sampler'),
      gl.getUniformLocation(conv1_2_program, 'layer3Sampler'),
      gl.getUniformLocation(conv1_2_program, 'layer4Sampler'),
    ],
    textures: [conv1_1_texture1, conv1_1_texture2, conv1_1_texture3, conv1_1_texture4],
    filters: [gl.NEAREST, gl.NEAREST, gl.NEAREST, gl.NEAREST],
    weights: get_conv1_2_weights(options.weights),
    biases: get_conv1_biases(options.weights),
    videoRes: videoRes
  };

  console.log("conv2_1 program");
  const conv2_1_program = init_conv2_1_program(gl, 644, 290);
  const conv2_1_program_info = {
    program: conv2_1_program,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(conv2_1_program, 'aVertexPosition'),
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(conv2_1_program, 'uProjectionModelViewMatrix'),
      weightsLocation: gl.getUniformLocation(conv2_1_program, 'weights'),
      videoResLocation: gl.getUniformLocation(conv2_1_program, 'videoRes')
    },
    samplers: [
      gl.getUniformLocation(conv2_1_program, 'layer1Sampler'),
      gl.getUniformLocation(conv2_1_program, 'layer2Sampler'),
      gl.getUniformLocation(conv2_1_program, 'layer3Sampler'),
      gl.getUniformLocation(conv2_1_program, 'layer4Sampler')
    ],
    textures: [conv1_2_texture1, conv1_2_texture2, conv1_2_texture3, conv1_2_texture4],
    filters: [gl.NEAREST, gl.NEAREST, gl.NEAREST, gl.NEAREST],
    weights: get_conv2_1_weights(options.weights),
    videoRes: videoRes
  };

  console.log("conv2_2 program");
  const conv2_2_program = init_conv2_2_program(gl, 644, 288);
  const conv2_2_program_info = {
    program: conv2_2_program,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(conv2_2_program, 'aVertexPosition'),
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(conv2_2_program, 'uProjectionModelViewMatrix'),
      weightsLocation: gl.getUniformLocation(conv2_2_program, 'weights'),
      biasesLocation: gl.getUniformLocation(conv2_2_program, 'biases'),
      videoResLocation: gl.getUniformLocation(conv2_2_program, 'videoRes')
    },
    samplers: [
      gl.getUniformLocation(conv2_2_program, 'layer1Sampler'),
      gl.getUniformLocation(conv2_2_program, 'layer2Sampler')
    ],
    textures: [conv2_1_texture1, conv2_1_texture2],
    filters: [gl.NEAREST, gl.NEAREST],
    weights: get_conv2_2_weights(options.weights),
    biases: get_conv2_biases(options.weights),
    videoRes: videoRes
  };

  console.log("reconstruct program");
  const reconstruct_program = init_reconstruct_program(gl, 642, 288);
  const reconstruct_program_info = {
    program: reconstruct_program,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(reconstruct_program, 'aVertexPosition'),
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(reconstruct_program, 'uProjectionModelViewMatrix'),
      weightsLocation: gl.getUniformLocation(reconstruct_program, 'weights'),
      biasesLocation: gl.getUniformLocation(reconstruct_program, 'biases'),
      videoResLocation: gl.getUniformLocation(reconstruct_program, 'videoRes')
    },
    samplers: [
      gl.getUniformLocation(reconstruct_program, 'originalSampler'),
      gl.getUniformLocation(reconstruct_program, 'layer1Sampler'),
      gl.getUniformLocation(reconstruct_program, 'layer2Sampler'),
    ],
    textures: [input_texture, conv2_2_texture1, conv2_2_texture2],
    weights: get_reconstruct_weights(options.weights),
    filters: [gl.LINEAR, gl.NEAREST, gl.NEAREST],
    biases: get_reconstruct_biases(options.weights),
    rgbBiases: true,
    videoRes: videoRes
  };

  console.log('Render program');
  const render_program = initRenderProgram(gl);
  const render_program_info = {
    program: render_program,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(render_program, 'aVertexPosition')
    },
    uniformLocations: {
      projectionModelViewMatrix: gl.getUniformLocation(render_program, 'uProjectionModelViewMatrix'),
      renderAreaLocation: gl.getUniformLocation(render_program, 'renderArea'),
      videoResLocation: gl.getUniformLocation(render_program, 'videoRes')
    },
    samplers: [gl.getUniformLocation(render_program, 'originalSampler')],
    textures: [reconstruct_texture],
    filters: [gl.LINEAR],
    renderArea: renderArea,
    videoRes: videoRes
  };

  player.on("loadedmetadata", () => {
    console.log("Video res:", video.videoWidth, video.videoHeight);
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;

    // Re-create all textures

    // Padded input
    pad_texture = createTexture(gl, videoWidth + 8, videoHeight + 8, true);

    conv1_1_texture1 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);
    conv1_1_texture2 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);
    conv1_1_texture3 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);
    conv1_1_texture4 = createTexture(gl, videoWidth + 8, videoHeight + 4, true);

    conv1_2_texture1 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);
    conv1_2_texture2 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);
    conv1_2_texture3 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);
    conv1_2_texture4 = createTexture(gl, videoWidth + 4, videoHeight + 4, true);

    conv2_1_texture1 = createTexture(gl, videoWidth + 4, videoHeight + 2, true);
    conv2_1_texture2 = createTexture(gl, videoWidth + 4, videoHeight + 2, true);

    conv2_2_texture1 = createTexture(gl, videoWidth + 2, videoHeight + 2, true);
    conv2_2_texture2 = createTexture(gl, videoWidth + 2, videoHeight + 2, true);

    reconstruct_texture = createTexture(gl, videoWidth * 3, videoHeight * 3, false);

    // Update Texture References
    conv1_1_program_info.textures = [pad_texture];
    conv1_2_program_info.textures = [conv1_1_texture1, conv1_1_texture2, conv1_1_texture3, conv1_1_texture4];
    conv2_1_program_info.textures = [conv1_2_texture1, conv1_2_texture2, conv1_2_texture3, conv1_2_texture4];
    conv2_2_program_info.textures = [conv2_1_texture1, conv2_1_texture2];
    reconstruct_program_info.textures = [input_texture, conv2_2_texture1, conv2_2_texture2];
    render_program_info.textures = [reconstruct_texture];
  });

  // Here's where we call the routine that builds all the
  // objects we'll be drawing.
  const buffers = initBuffers(gl);

  // Create and bind the framebuffer
  const pad_fb = gl.createFramebuffer();
  const w_conv1_1_fb = gl.createFramebuffer();
  const w_conv1_2_fb = gl.createFramebuffer();
  const w_conv2_1_fb = gl.createFramebuffer();
  const w_conv2_2_fb = gl.createFramebuffer();
  const w_reconstruct_fb = gl.createFramebuffer();

  let elapsedTime = 0;
  let frameCount = 0;
  let lastTime = new Date().getTime();
  let fps = 0;
  let frameDelay = 0;

  /**
   * Draw the scene repeatedly
   */
  function render(now) {
    if (!copyVideo) return;

    updateTexture(gl, input_texture, video);
    resizeCanvas(canvas);

    const renderSettings = scaleToFit(videoWidth, videoHeight, canvas.width, canvas.height);
    // console.log("renderSettings:", renderSettings);

    padProgramInfo.videoRes = [videoWidth, videoHeight];
    conv1_1_program_info.videoRes = [videoWidth, videoHeight];
    conv1_2_program_info.videoRes = [videoWidth, videoHeight];
    conv2_1_program_info.videoRes = [videoWidth, videoHeight];
    conv2_2_program_info.videoRes = [videoWidth, videoHeight];
    reconstruct_program_info.videoRes = [videoWidth, videoHeight];

    render_program_info.videoRes = renderSettings[0];
    render_program_info.renderArea = renderSettings[1];


    // PAD INPUT
    gl.bindFramebuffer(gl.FRAMEBUFFER, pad_fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pad_texture, 0);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0, // gl_FragData[0]
    ]);
    gl.viewport(0, 0, videoWidth + 8, videoHeight + 8);
    drawScene(gl, padProgramInfo, buffers);


    // Apply W_conv1_1
    gl.bindFramebuffer(gl.FRAMEBUFFER, w_conv1_1_fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, conv1_1_texture1, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, conv1_1_texture2, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, conv1_1_texture3, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, conv1_1_texture4, 0);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3
    ]);
    gl.viewport(0, 0, videoWidth + 8, videoHeight + 4);
    drawScene(gl, conv1_1_program_info, buffers);


    // Apply W_conv1_2, relu and bias
    // in: 648x290x8
    // out: 644x290x8
    gl.bindFramebuffer(gl.FRAMEBUFFER, w_conv1_2_fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, conv1_2_texture1, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, conv1_2_texture2, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, conv1_2_texture3, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, conv1_2_texture4, 0);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
      gl.COLOR_ATTACHMENT2,
      gl.COLOR_ATTACHMENT3,
    ]);
    gl.viewport(0, 0, videoWidth + 4, videoHeight + 4);
    drawScene(gl, conv1_2_program_info, buffers);


    // Apply W_conv2_1
    // in: 644x290x8
    // out: 644x288x4
    gl.bindFramebuffer(gl.FRAMEBUFFER, w_conv2_1_fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, conv2_1_texture1, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, conv2_1_texture2, 0);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
    ]);
    gl.viewport(0, 0, videoWidth + 4, videoHeight + 2);
    drawScene(gl, conv2_1_program_info, buffers);


    // Apply W_conv2_2, relu and bias
    // in: 644x288x4
    // out: 642x288x4
    gl.bindFramebuffer(gl.FRAMEBUFFER, w_conv2_2_fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, conv2_2_texture1, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, conv2_2_texture2, 0);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
    ]);
    gl.viewport(0, 0, videoWidth + 2, videoHeight + 2);
    drawScene(gl, conv2_2_program_info, buffers);


    // Reconstruct
    // in: 642x288x4
    // out: 640x286x27
    // out: 1920x858x3 (hard)
    // Scale the current texture
    // Sum and clamp
    gl.bindFramebuffer(gl.FRAMEBUFFER, w_reconstruct_fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reconstruct_texture, 0);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0, // gl_FragData[0]
    ]);
    gl.viewport(0, 0, videoWidth * 3, videoHeight * 3);
    drawScene(gl, reconstruct_program_info, buffers);


    // Render Final Video
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    drawScene(gl, render_program_info, buffers);

    now = new Date().getTime();
    frameCount++;
    elapsedTime += now - lastTime;

    lastTime = now;
    if (elapsedTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      elapsedTime -= 1000;

      // frameDelay minimum of 1 avoids the frameDelay getting stuck at zero in the update expression below
      if (frameDelay < 1) {
        frameDelay = 1
      }
      // set frameDelay based on the current value and how close we are to targetFrameRate
      frameDelay = frameDelay * fps / targetFrameRate;

      console.log("fps", fps);
    }

    // Do it again!
    requestAnimationFrame(() => setTimeout(render, frameDelay));
  }

  requestAnimationFrame(render);
}
