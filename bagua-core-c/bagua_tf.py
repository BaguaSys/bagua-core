import tensorflow as tf
import bagua.torch_api as bagua
from bagua.bagua_define import DistributedAlgorithm

bagua_allreduce = tf.load_op_library("./bagua_allreduce.so")


def _make_allreduce_grads_fn(name):
    def allreduce_grads(grads):
        nccl_unique_id_str = bagua.communication.gen_nccl_unique_id(
            "way", root=0)

        with tf.name_scope(name + "_Allreduce"):
            result = []
            for grad in grads:
                if grad is None:
                    result.append(grad)
                    continue

                result += bagua_allreduce.grouped_allreduce(
                    tensors=[grad],
                    rank=bagua.get_rank(),
                    nranks=bagua.get_world_size(),
                    device_id=bagua.get_local_rank(),
                    nccl_unique_id_str=nccl_unique_id_str,
                    comm_tensors_num=len(grads))

            return result

            return [bagua_allreduce.grouped_allreduce(
                tensors=[grad],
                rank=bagua.get_rank(),
                nranks=bagua.get_world_size(),
                device_id=bagua.get_local_rank(),
                nccl_unique_id_str=nccl_unique_id_str,
                comm_tensors_num=len(grads)) if grad is not None else grad for grad in grads]

            # return [_allreduce_cond(grad,
            #                         device_dense=device_dense,
            #                         device_sparse=device_sparse,
            #                         compression=compression,
            #                         op=op,
            #                         prescale_factor=prescale_factor,
            #                         postscale_factor=postscale_factor)
            #         if grad is not None else grad
            #         for grad in grads]

            # # TODO: 分桶处理
            # return bagua_allreduce.grouped_allreduce(
            #     tensors=grads,
            #     rank=bagua.get_rank(),
            #     nranks=bagua.get_world_size(),
            #     device_id=bagua.get_local_rank(),
            #     nccl_unique_id_str=nccl_unique_id_str,
            #     comm_tensors_num=len(grads),
            # )

    return allreduce_grads


class DistributeOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer,
                 distributed_algorithm=DistributedAlgorithm.GradientAllReduce,
                 name=None, use_locking=False):
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)
        super(DistributeOptimizer, self).__init__(
            name=name, use_locking=use_locking)

        self._optimizer = optimizer
        self._distributed_algorithm = distributed_algorithm
        self._allreduce_grads = _make_allreduce_grads_fn(name)

        self._agg_helper = None
        self.iter = 0

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        grads, vars = zip(*gradients)

        # # Decentralize =>
        # if self._distributed_algorithm == DistributedAlgorithm.Decentralize:
        #     avg_vars = self._allreduce_grads()
        #     return list(zip(avg_vars, vars))

        avg_grads = self._allreduce_grads(grads)
        return list(zip(avg_grads, vars))

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        if self._agg_helper:
            return self._agg_helper.apply_gradients(
                lambda: self._optimizer.apply_gradients(*args, **kwargs),
                self._optimizer,
                *args,
                **kwargs,
            )

        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)
