"""Conservative style-only edits: preserve required calculations, no line limit."""
VERSION = "maas_concise_v2"

ENDING = "\nKeep every calculation needed to obtain the answer. Use short equations and brief explanations instead of restating the question or writing a tutorial. Do not skip dependencies or invent missing values to shorten the solution. Verify the result once. State the final answer once, using the task's requested final-answer format, then stop; do not restart or repeat the solution.\n"


def install(instructions, operators):
    for name in ("GENERATE_SOLUTION_PROMPT", "REFINE_ANSWER_PROMPT", "SOLUTION_PROMPT", "MATH_SOLUTION_PROMPT", "MATH_SOLVE_PROMPT", "DETAILED_SOLUTION_PROMPT"):
        text=getattr(instructions,name)
        # Keep the native order of calculation, checking, and final answer.
        lines=[line for line in text.splitlines() if not any(s in line for s in (
            "State the problem clearly", "clear restatement", "Begin with a clear statement",
            "Visual aids or diagrams", "significance of the result", "educational for someone"))]
        text="\n".join(lines)
        for old,new in (("comprehensive, step-by-step", "complete, step-by-step"),
                        ("well-formatted and detailed", "well-formatted and concise"),
                        ("detailed calculations", "necessary calculations"),
                        ("detailed, logical progression", "complete, logical progression"),
                        ("detailed,", "concise,"),
                        ("thorough, mathematically sound", "concise, mathematically sound"),
                        ("comprehensive, mathematically rigorous", "concise, mathematically rigorous")):
            text=text.replace(old,new)
        setattr(instructions,name,text+ENDING)
    operators.GENERATE_COT_PROMPT += ENDING
    operators.PYTHON_CODE_VERIFIER_PROMPT += "\nReturn one fenced Python code block containing all necessary calculations in solve(). Do not add prose, a walkthrough, example execution, print statements, or duplicate code. Stop after the code block.\n"
    operators.SELFREFINE_PROMPT += ENDING
    operators.SC_ENSEMBLE_PROMPT = operators.SC_ENSEMBLE_PROMPT.replace(
        "provide a detailed explanation of your thought process", "give one short sentence explaining the selection")
