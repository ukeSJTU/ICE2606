# Makefile for converting Markdown reports to DOCX using Pandoc for multiple labs

.PHONY: all clean lab1 lab2 lab3 lab4

all: lab1 lab2 lab3 lab4

lab1:
	if [ -f Lab1/report.md ]; then \
		cd Lab1 && pandoc -s report.md -o report.docx; \
		echo "Converted Lab1/report.md to Lab1/report.docx"; \
	else \
		echo "Lab1/report.md not found, skipping..."; \
	fi

lab2:
	if [ -f Lab2/report.md ]; then \
		cd Lab2 && pandoc -s report.md -o report.docx; \
		echo "Converted Lab2/report.md to Lab2/report.docx"; \
	else \
		echo "Lab2/report.md not found, skipping..."; \
	fi

lab3:
	if [ -f Lab3/report.md ]; then \
		cd Lab3 && pandoc -s report.md -o report.docx; \
		echo "Converted Lab3/report.md to Lab3/report.docx"; \
	else \
		echo "Lab3/report.md not found, skipping..."; \
	fi

lab4:
	if [ -f Lab4/report.md ]; then \
		cd Lab4 && pandoc -s report.md -o report.docx; \
		echo "Converted Lab4/report.md to Lab4/report.docx"; \
	else \
		echo "Lab4/report.md not found, skipping..."; \
	fi

clean:
	for lab in Lab1 Lab2 Lab3 Lab4; do \
		if [ -f $$lab/report.docx ]; then \
			rm $$lab/report.docx; \
			echo "Removed $$lab/report.docx"; \
		fi; \
	done
