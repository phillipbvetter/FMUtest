all: test_model.so
	#Comment here preserves the prior tab
test_model.so: test_model.cpp
	Rscript --vanilla -e "TMB::compile(file='test_model.cpp', flags='-O0 -g', framework = 'TMBad')"
clean:
	rm -rf *o