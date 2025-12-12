from mpi4py import MPI
import numpy as np


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        # Get_size() gives the world size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert (
            src_array.size % nprocs == 0
        ), "src_array size must be divisible by the number of processes"
        assert (
            dest_array.size % nprocs == 0
        ), "dest_array size must be divisible by the number of processes"

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.

        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        # Get_size() gives the world size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        # "Manual" all-reduce: reduce, then bcast
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        # Step 1: allocate receive buffer on root, then sync
        if rank == 0:
            all_arrays = np.empty((nprocs, src_array.size), dtype=src_array.dtype)
        else:
            all_arrays = None
        self.Barrier()

        # Step 2: send all arrays to root
        if rank == 0:
            all_arrays[0, :] = src_array
            for i in range(1, nprocs):
                self.comm.Recv(all_arrays[i], source=i, tag=123)
        else:
            self.comm.Send(src_array, dest=0, tag=123)
        self.Barrier()

        # Step 3: reduce on root then broadcast result
        if rank == 0:
            if op == MPI.SUM:
                dest_array[:] = np.sum(all_arrays, axis=0)
            elif op == MPI.PROD:
                dest_array[:] = np.prod(all_arrays, axis=0)
            elif op == MPI.MIN:
                dest_array[:] = np.min(all_arrays, axis=0)
            elif op == MPI.MAX:
                dest_array[:] = np.max(all_arrays, axis=0)
            else:
                raise ValueError("Unsupported op in myAllreduce")
        self.comm.Bcast(dest_array, root=0)

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.

        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.

        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        seg_size = src_array.size // nprocs

        # Sanity check: ensure arrays can be evenly divided among procs
        assert src_array.size % nprocs == 0
        assert dest_array.size % nprocs == 0
        assert src_array.shape == dest_array.shape

        # src_array is assumed to be a 1D (flattened) numpy array
        for i in range(nprocs):
            src_offset = i * seg_size
            dest_offset = i * seg_size
            if i == rank:
                # Copy local segment directly
                dest_array[dest_offset : dest_offset + seg_size] = src_array[
                    src_offset : src_offset + seg_size
                ]
                # No bytes transferred for local copy
            else:
                # Prepare send and recv buffers
                sendbuf = np.ascontiguousarray(
                    src_array[src_offset : src_offset + seg_size]
                )
                recvbuf = np.empty(seg_size, dtype=src_array.dtype)
                # Sendrecv: exchange segment with process i
                self.comm.Sendrecv(
                    sendbuf, dest=i, sendtag=456, recvbuf=recvbuf, source=i, recvtag=456
                )
                dest_array[dest_offset : dest_offset + seg_size] = recvbuf
                # Update bytes transferred: both send and recv for each segment
                self.total_bytes_transferred += seg_size * src_array.itemsize * 2
