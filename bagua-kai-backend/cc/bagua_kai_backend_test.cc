#include <vector>
#include <iostream>

#include <gtest/gtest.h>
#include <cuda.h>
#include <sys/wait.h>
#include <unistd.h>

#include "bagua_kai_backend.h"

TEST(BaguaKaiBackend, EndToEnd)
{
    std::vector<std::vector<int> > processes_gpu_setting{
        {0}, {1, 2}, {3, 4, 5, 6, 7}};

    std::vector<int> child_pids;
    for (std::vector<int> gpu_setting : processes_gpu_setting)
    {
        pid_t c_pid = fork();
        ASSERT_NE(c_pid, -1);

        if (c_pid > 0) {
            // parent process
            child_pids.push_back(c_pid);
        } else {
            // child process
            // Expect two strings not to be equal.
            EXPECT_STRNE("hello", "world");
            // Expect equality.
            EXPECT_EQ(7 * 6, 42);
            std::cerr << "Hello from " << getpid() << std::endl;
            exit(EXIT_SUCCESS);
        }
    }

    // wait child
    wait(nullptr);
}
