#version 330


uniform sampler2D tex;


smooth in vec2 f_texCoods;

layout(location=0) out vec4 out_color;


void main()
{
	out_color = texture(tex, f_texCoods);
}
