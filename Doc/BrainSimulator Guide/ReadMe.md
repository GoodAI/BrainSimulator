* Built with [MkDocs](http://www.mkdocs.org/)
* Using [Markdown syntax](http://daringfireball.net/projects/markdown/syntax)
* Supports [MathJax](https://www.mathjax.org/)
	* MathJax configuration in docs/js/mathjaxhelper.js

Changelog
---------
* 30.6.2015 - Created GoodAI theme
* 29.6.2015 - Added support for [Mathjax](https://www.mathjax.org/) -- now you can write equations -- e.g. $$ a = b^3 $$. Inline equations are also supported -- e.g. $$ a = b^3 $$

How to edit documentation
-------------------------
* Edit *.md files in ./docs folder
* Build HTML files by running: `mkdocs build`
* Show documentation locally by running: `mkdocs serve`

How to use MkDocs
-----------------
1. Install [Python](https://www.python.org/downloads/) - you can use Python 3 (tested with 3.5)
2. (Install [pip](http://pip.readthedocs.org/en/latest/installing.html)) - standard windows installer installs also pip
3. Install MkDocs - `pip install mkdocs` (you may need to run the command prompt as the administrator, also go to _Python_dir_\Scripts if the installer does not update the paths)
4. Serve locally - `mkdocs serve`
5. Build HTML files - `mkdocs build` as ...Docs\BrainSimulator Guide>mkdocs build (running mkdocs with abs path _Python_dir_\Scripts\mkdocs should not be necessary)

GoodAI theme
------------
* Design is based on default MkDocs theme (which is based on [Bootstrap](http://getbootstrap.com/))
Most of the webpage styling happens in ./goodai_theme/css/bootstrap-custom.min.css.
* Custom Bootstrap CSS was created using [Lavish Bootstrap](http://www.lavishbootstrap.com/) tool and GoodAI logo (light version) colors

Useful tools and resources
------------
* Recording screen as GIF - https://screentogif.codeplex.com/
* Screenshots - http://getgreenshot.org/
* LaTeX to PNG - http://latex2png.com/

* Markdown cheatsheet - https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

