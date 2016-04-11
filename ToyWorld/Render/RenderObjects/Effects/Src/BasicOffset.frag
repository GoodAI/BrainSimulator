#version 330

uniform sampler2D tex;

in vec2 f_texCoods;

layout(location=0) out vec4 out_color;

void main()
{
	//out_color = texture(tex, f_texCoods);
	out_color = vec4(f_texCoods.xxx, 1);
	//out_color = vec4(1,1,1,1);
}
