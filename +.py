import glfw
import numpy as np
from OpenGL.GL import *
import time


# Функция для компиляции шейдера
def compileShader(shaderType, src):
    shader = glCreateShader(shaderType)
    glShaderSource(shader, src)
    glCompileShader(shader)

    # Проверяем статус компиляции
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"Error compiling shader:\n{error}")
        glDeleteShader(shader)
        return None

    return shader

# Функция для создания программы шейдеров
def createShader(vertex, fragment):
    vertexShader = compileShader(GL_VERTEX_SHADER, vertex)
    fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragment)

    # Если компиляция не удалась, завершаем программу
    if not vertexShader or not fragmentShader:
        return None

    program = glCreateProgram()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glLinkProgram(program)

    # Проверяем статус линковки программы
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        print(f"Error linking program:\n{error}")
        glDeleteProgram(program)
        return None

    glValidateProgram(program)

    glDeleteShader(vertexShader)
    glDeleteShader(fragmentShader)

    return program

def main():
    # Инициализация GLFW
    if not glfw.init():
        return

    # Создание окна GLFW
    window = glfw.create_window(640, 480, "Fractal Pyramid", None, None)
    if not window:
        glfw.terminate()
        return

    # Устанавливаем контекст окна текущим
    glfw.make_context_current(window)

    # Определение вершинных данных
    vertices = np.array([
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        -1.0, 1.0, 0.0,
        -1.0, -1.0, 0.0,
    ], dtype=np.float32)

    # Создание буфера вершин (VBO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Чтение и компиляция шейдеров
    with open('vertex.shader', 'r', encoding='utf-8') as file:
        vertex_shader_source = file.read()
    with open('fragment.shader', 'r', encoding='utf-8') as file:
        fragment_shader_source = file.read()

    # Создание и использование программы шейдеров
    shader_program = createShader(vertex_shader_source, fragment_shader_source)
    if not shader_program:
        glfw.terminate()
        return

    glUseProgram(shader_program)

    # Указываем атрибуты вершин
    position_loc = glGetAttribLocation(shader_program, "position")
    glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
    glEnableVertexAttribArray(position_loc)

    # Получаем местоположения uniform-переменных
    iResolution_loc = glGetUniformLocation(shader_program, "iResolution")
    iTime_loc = glGetUniformLocation(shader_program, "iTime")

    # Основной цикл рендеринга
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # Устанавливаем значения uniform-переменных
        width, height = glfw.get_framebuffer_size(window)
        glUniform2f(iResolution_loc, width, height)
        glUniform1f(iTime_loc, glfw.get_time())  # Передаем текущее время

        # Рендеринг прямоугольника
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Обновление окна
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Завершение работы GLFW
    glfw.terminate()


if __name__ == "__main__":
    main()
