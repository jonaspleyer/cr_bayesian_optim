CC:=latexmk
OPTIONS:=-pdf
TARGET:=main

all:
	$(CC) $(OPTIONS) $(TARGET)

clean_partial:
	rm -f $(TARGET).aux
	rm -f $(TARGET).bbl
	rm -f $(TARGET).blg
	rm -f $(TARGET).dvi
	rm -f $(TARGET).fdb_latexmk
	rm -f $(TARGET).log
	rm -f $(TARGET).out
	rm -f $(TARGET).fls
	rm -f archive.zip

clean: clean_partial
	rm -f $(TARGET).pdf

fresh: clean all

zip: fresh
	$(MAKE) clean_partial
	zip archive.zip -r *
