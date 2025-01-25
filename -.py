import glfw
from OpenGL.GL import *
import numpy as np

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

fragment_shader_source = """
#version 330 core

out vec4 fragColor;

uniform float iTime;
uniform vec2 iResolution;

const float cloudscale = 1.1;
const float speed = 0.03;
const float clouddark = 0.5;
const float cloudlight = 0.3;
const float cloudcover = 0.2;
const float cloudalpha = 8.0;
const float skytint = 0.5;
const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
const vec3 skycolour2 = vec3(0.4, 0.7, 1.0);

const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

vec2 hash( vec2 p ) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
    vec2 i = floor(p + (p.x + p.y) * K1);    
    vec2 a = p - i + (i.x + i.y) * K2;
    vec2 o = (a.x > a.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0); 
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash(i + 0.0)), dot(b, hash(i + o)), dot(c, hash(i + 1.0)));
    return dot(n, vec3(70.0));    
}

float fbm(vec2 n) {
    float total = 0.0, amplitude = 0.1;
    for (int i = 0; i < 7; i++) {
        total += noise(n) * amplitude;
        n = m * n;
        amplitude *= 0.4;
    }
    return total;
}

void main() {
    vec2 p = gl_FragCoord.xy / iResolution.xy;
    vec2 uv = p * vec2(iResolution.x / iResolution.y, 1.0);    
    float time = iTime * speed;
    float q = fbm(uv * cloudscale * 0.5);
    
    // ridged noise shape
    float r = 0.0;
    uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i = 0; i < 8; i++){
        r += abs(weight * noise(uv));
        uv = m * uv + time;
        weight *= 0.7;
    }
    
    // noise shape
    float f = 0.0;
    uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i = 0; i < 8; i++){
        f += weight * noise(uv);
        uv = m * uv + time;
        weight *= 0.6;
    }
    
    f *= r + f;
    
    // noise colour
    float c = 0.0;
    time = iTime * speed * 2.0;
    uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    uv *= cloudscale * 2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i = 0; i < 7; i++){
        c += weight * noise(uv);
        uv = m * uv + time;
        weight *= 0.6;
    }
    
    // noise ridge colour
    float c1 = 0.0;
    time = iTime * speed * 3.0;
    uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    uv *= cloudscale * 3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i = 0; i < 7; i++){
        c1 += abs(weight * noise(uv));
        uv = m * uv + time;
        weight *= 0.6;
    }
    
    c += c1;
    
    vec3 skycolour = mix(skycolour2, skycolour1, p.y);
    vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight * c), 0.0, 1.0);
   
    f = cloudcover + cloudalpha * f * r;
    
    vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
    
    fragColor = vec4(result, 1.0);
}
"""


def compileShader(shaderType, source):
    shader = glCreateShader(shaderType)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Check compilation status
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"Error compiling shader:\n{error}")
        glDeleteShader(shader)
        return None

    return shader


def createShader(vertex_source, fragment_source):
    vertex_shader = compileShader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = compileShader(GL_FRAGMENT_SHADER, fragment_source)

    if not vertex_shader or not fragment_shader:
        return None

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    # Check linking status
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(shader_program).decode()
        print(f"Error linking program:\n{error}")
        glDeleteProgram(shader_program)
        return None

    glValidateProgram(shader_program)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "Fractal Shader Example", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Define vertices and VBO (you can adjust as needed)
    vertices = np.array([
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        -1.0, 1.0, 0.0,
        -1.0, -1.0, 0.0
    ], dtype=np.float32)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Compile and create shader program
    shader_program = createShader(vertex_shader_source, fragment_shader_source)
    if not shader_program:
        glfw.terminate()
        return

    glUseProgram(shader_program)

    # Main render loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Set uniform variables
        glUniform1f(glGetUniformLocation(shader_program, "iTime"), glfw.get_time())
        width, height = glfw.get_framebuffer_size(window)
        glUniform2f(glGetUniformLocation(shader_program, "iResolution"), width, height)

        # Draw triangles
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Terminate GLFW
    glfw.terminate()


if __name__ == "__main__":
    main()
