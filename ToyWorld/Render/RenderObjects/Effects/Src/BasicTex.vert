#version 330

layout(location=0) in vec3 v_position;
layout(location=1) in vec2 v_texCoods;

out vec2 f_texCoods;

void main()
{
	f_texCoods = v_texCoods;
	gl_Position = vec4(v_position, 2);
}
