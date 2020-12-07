
def clean_tex(file):
    section = 0
    subsection = 0
    lincount = 0
    begcheck = False
    clfile = open('clfile.tex', 'w')

    with open(file, 'r') as f:
        for line in f:
            if '\\begin{' in line: # check for a \begin{ in line
                begcheck = True
                if '\\begin{document}' in line: # \begin{document} is in every file
                    begcheck = False
            if '\\end{' in line:
                begcheck = False
            if begcheck == False:
                clfile.write(line)

    f.close()
    clfile.close()


if __name__ == "__main__":
    clean_tex('report.tex')










