CC = g++

TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TARGETS=bagua_allreduce.so
cbind_lib = -L/share/ai_platform/changjianbin/playground/autotune_finetune/gossipy/bagua_comm_core/target/debug -l:

all: $(TARGETS)
clean:
	$(RM) $(TARGETS)

bagua_allreduce.so: bagua_allreduce.cc
	$(CC) -O2 -shared -std=c++11 -fPIC bagua_allreduce.cc \
		-I /usr/local/cuda-10.1/targets/x86_64-linux/include/ \
		-L/share/ai_platform/changjianbin/playground/autotune_finetune/gossipy/bagua_comm_core/target/debug/ -l bagua_comm_core_cbind \
		$(TF_CFLAGS) $(TF_LFLAGS) $(CFLAGS) \
		 -o $@
