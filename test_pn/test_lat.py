import subprocess
import os

def render_latex_to_image(latex_expression, filename, format="png", dpi=300):
    """
    Renders a LaTeX expression into an image file (PNG or SVG).
    Requires 'latex' and 'dvisvgm' or 'dvipng' to be installed and in PATH.

    Args:
        latex_expression (str): The LaTeX code to render.
        filename (str): The base name for the output image file.
        format (str): 'png' or 'svg'.
        dpi (int): Dots per inch for PNG output (ignored for SVG).
    """
    if format not in ["png", "svg"]:
        raise ValueError("Invalid format. Choose 'png' or 'svg'.")

    # Create a minimal LaTeX document for standalone equation/text
    # Using 'standalone' class is generally better for images as it crops tightly
    latex_document = f"""
\\documentclass[preview]{{standalone}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\begin{{document}}
{latex_expression}
\\end{{document}}
"""
    tex_filename = f"{filename}.tex"
    dvi_filename = f"{filename}.dvi"
    image_filename = f"{filename}.{format}"

    print(f"Attempting to render: {filename}.{format}")

    try:
        with open(tex_filename, "w") as f:
            f.write(latex_document)
        print(f"LaTeX file '{tex_filename}' created.")

        # Compile .tex to .dvi
        latex_process = subprocess.run(
            ["latex", "-interaction=nonstopmode", tex_filename],
            capture_output=True, text=True, check=False # Do not check=True here to allow custom error handling
        )
        if latex_process.returncode != 0:
            print(f"Error compiling '{tex_filename}' to '{dvi_filename}'.")
            print(f"Stdout: {latex_process.stdout}")
            print(f"Stderr: {latex_process.stderr}")
            print(f"Check '{filename}.log' for LaTeX errors.")
            return # Exit if latex compilation fails

        print(f"Successfully compiled '{tex_filename}' to '{dvi_filename}'.")

        if format == "png":
            # Convert .dvi to .png using dvipng
            subprocess.run(
                ["dvipng", "-D", str(dpi), "-o", image_filename, dvi_filename],
                capture_output=True, text=True, check=True
            )
        elif format == "svg":
            # Convert .dvi to .svg using dvisvgm
            subprocess.run(
                ["dvisvgm", "-o", image_filename, dvi_filename],
                capture_output=True, text=True, check=True
            )
        print(f"Successfully converted '{dvi_filename}' to '{image_filename}'.")

    except FileNotFoundError as e:
        print(f"Error: Command '{e.filename}' not found.")
        print("Please ensure you have LaTeX (latex) and either dvipng or dvisvgm installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error during image conversion:")
        print(f"Command: {e.cmd}")
        print(f"Return Code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up auxiliary files
        for ext in [".aux", ".log", ".dvi"]:
            if os.path.exists(f"{filename}{ext}"):
                os.remove(f"{filename}{ext}")
        print(f"Cleaned up auxiliary files for '{filename}'.")
        print("-" * 30) # Separator for clarity


if __name__ == "__main__":
    # Collection of LaTeX formulas to be rendered
    latex_formulas = {
        "derivative_e_power_x_squared": r"""
$y = e^{-x^2+2x+1} \implies \frac{dy}{dx} = e^{-x^2+2x+1} \frac{d}{dx}(-x^2+2x+1)$
$= e^{-x^2+2x+1} (-2x+2)$

$\frac{d^2y}{dx^2} = (e^{-x^2+2x+1} (-2x+2))'$
$= (e^{-x^2+2x+1})' (-2x+2) + e^{-x^2+2x+1} (-2x+2)'$
$= e^{-x^2+2x+1} (-2x+2)^2 + e^{-x^2+2x+1} (-2)$
$= e^{-x^2+2x+1} ((2-2x)^2 - 2)$
""",
        "derivative_ln_x": r"""
$y = \ln x \implies \frac{dy}{dx} = \frac{1}{x} \implies \frac{d^2y}{dx^2} = -\frac{1}{x^2}$
""",
        "derivative_quotient_rule": r"""
$y = \frac{x+1}{x^2+1} \implies \frac{dy}{dx} = \frac{(x^2+1)\frac{d}{dx}(x+1) - (x+1)\frac{d}{dx}(x^2+1)}{(x^2+1)^2}$
$= \frac{x^2+1 - (x+1)(2x)}{(x^2+1)^2}$
$= \frac{x^2+1-2x^2-2x}{(x^2+1)^2} = \frac{1-x^2-2x}{(x^2+1)^2}$
$\implies \frac{d^2y}{dx^2} = \frac{2x^3+6x^2-6x-2}{(x^2+1)^3}$
""",
        "sum_of_derivatives_limit_definition": r"""
$(f+g)'(x_o) = \lim_{x \to x_o} \frac{(f+g)(x)-(f+g)(x_o)}{x-x_o}$
$= \lim_{x \to x_o} \frac{f(x)+g(x)-f(x_o)-g(x_o)}{x-x_o}$
$= \lim_{x \to x_o} \left(\frac{f(x)-f(x_o)}{x-x_o} + \frac{g(x)-g(x_o)}{x-x_o}\right)$
$= \lim_{x \to x_o} \left(\frac{f(x)-f(x_o)}{x-x_o}\right) + \lim_{x \to x_o} \left(\frac{g(x)-g(x_o)}{x-x_o}\right)$
$= f'(x_o) - g'(x_o)$
""",
        "proof_derivative_sin_x": r"""
$f(x) = \sin x \implies f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h} = \lim_{h \to 0} \frac{\sin(x+h)-\sin x}{h}$
$\implies f'(x) = \lim_{h \to 0} \frac{\sin x \cos h + \cos x \sin h - \sin x}{h}$
$= \lim_{h \to 0} \left(\frac{\sin x (\cos h - 1)}{h} + \frac{\cos x \sin h}{h}\right)$
$= \sin x \lim_{h \to 0} \frac{\cos h - 1}{h} + \cos x \lim_{h \to 0} \frac{\sin h}{h} = (\sin x) \times 0 + (\cos x) \times 1 = \cos x.$
""",
        "proof_derivative_cos_x": r"""
$f(x) = \cos x \implies f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h} = \lim_{h \to 0} \frac{\cos(x+h)-\cos x}{h}$
$= \lim_{h \to 0} \frac{\cos x \cos h - \sin x \sin h - \cos x}{h}$
$= \lim_{h \to 0} \left(\frac{\cos x (\cos h - 1)}{h} - \frac{\sin x \sin h}{h}\right)$
$= \cos x \lim_{h \to 0} \frac{\cos h - 1}{h} - \sin x \lim_{h \to 0} \frac{\sin h}{h} = (\cos x) \times 0 - (\sin x) \times 1 = -\sin x$
"""
    }

    # Render each formula
    # for name, latex_code in latex_formulas.items():
    #     render_latex_to_image(latex_code, filename=name, format="png", dpi=600)
        # You can also render as SVG if desired:
        # render_latex_to_image(latex_code, filename="hello", format="svg")
#render_latex_to_image("hello", filename="hello", format="png", dpi=800)
