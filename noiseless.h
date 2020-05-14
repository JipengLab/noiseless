#pragma once

#include <time.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <thread>
#include <optional>
#include <type_traits>
#include <future>
#include <functional>
#include <chrono>
#include <limits>
#include <cmath>
#include <sstream>
#include <vector>

namespace noiseless
{

  typedef uint8_t u8;
  typedef uint16_t u16;
  typedef uint32_t u32;
  typedef uint64_t u64;
  typedef int8_t i8;
  typedef int16_t i16;
  typedef int32_t i32;
  typedef int64_t i64;
  typedef float f32;
  typedef double f64;

  inline auto GenStr = [](auto&&... item) {
    std::stringstream s;
    (s << ... << item);
    return s.str();
  };

  struct RepeatedLatencies
  {
    std::vector<f64> latencies;
    f64 mean;
    f64 stddev;
    f64 min = std::numeric_limits<f64>::min();
    f64 max = std::numeric_limits<f64>::max();
    std::string ToString(){
      std::stringstream ss;
      ss << "mean(ms) = " << mean <<", stddev(ms) = " << stddev << ", min(ms) = " << min << ", max(ms) = " << max << "\nlatencies(ms): ";
      for(auto latency : latencies){
        ss << latency << " ";
      }
      ss << std::endl;
      return ss.str();
    }
  };

  class Noiseless
  {
  public:
    static Noiseless &Instance()
    {
      static Noiseless instance;
      return instance;
    }
    static const int kFirstInFirstOut = SCHED_FIFO;
    static const int kRoundRobin = SCHED_RR;
    static const int kOther = SCHED_OTHER;

    static timespec Now()
    {
      timespec tp;
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tp);
      return tp;
    }

    static u64 MicrosecondsElapsed(const timespec &tp)
    {
      timespec now;
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
      u64 microsec = (now.tv_sec - tp.tv_sec) * 1000000;
      microsec += (now.tv_nsec - tp.tv_nsec) / 1000;
      return microsec;
    }

    static f64 MillisecondsElapsed(const timespec &tp)
    {
      return (f64) MicrosecondsElapsed(tp) / 1000;
    }

    int MaxPriority(int policy)
    {
      return sched_get_priority_max(policy);
    }

    int MinPriority(int policy)
    {
      return sched_get_priority_min(policy);
    }

    void SetSchedPolicyFor(std::thread tid, int policy, int priority)
    {
      sched_param sch;
      sch.sched_priority = priority;
      if (pthread_setschedparam(tid.native_handle(), policy, &sch) != 0)
      {
        throw std::runtime_error(strerror(errno));
      }
    }

    void SetSchedPolicyForCurrentThread(int policy, int priority)
    {
      sched_param sch;
      sch.sched_priority = priority;
      if (pthread_setschedparam(pthread_self(), policy, &sch) != 0)
      {
        throw std::runtime_error(strerror(errno));
      }
    }

    void SetCpuAffinityFor(std::thread tid, int cpuid)
    {
      if (cpuid >= nr_cpus_)
        throw std::runtime_error("invalid cpuid");
      if (pthread_setaffinity_np(tid.native_handle(), sizeof(cpu_set_t), &cpu_set_) != 0)
      {
        throw std::runtime_error(strerror(errno));
      }
    }

    void SetCpuAffinityForCurrentThread(int cpuid)
    {
      if (cpuid >= nr_cpus_)
        throw std::runtime_error("invalid cpuid");
      if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set_) != 0)
      {
        throw std::runtime_error(strerror(errno));
      }
    }

    void SetCpuAffinityToNotIsolatedCpu(){
      SetCpuAffinityForCurrentThread(0); 
      std::this_thread::sleep_for(std::chrono::milliseconds(200)); // pause 200 ms
    }

    template <typename F, typename... Args>
    std::invoke_result_t<F, Args...> RunWithoutNoiseless(f64 &ms_elapsed, F &&f,
                                                         Args &&... args)
    {
      auto start = Now();
      std::invoke_result_t<F, Args...> tmp = std::invoke(std::forward<F>(f), 
                                                         std::forward<Args>(args)...);
      ms_elapsed = (f64)MicrosecondsElapsed(start) / 1000;
      return tmp;
    }

    template <typename F, typename... Args>
    std::invoke_result_t<F, Args...> Run(f64 &ms_elapsed, F &&f,
                                         Args &&... args)
    {
      auto promise =
          std::make_shared<std::promise<std::invoke_result_t<F, Args...>>>();
      auto future = promise->get_future();
      int cpuid = isolated_cpu_id_;
      std::thread tid([this, &ms_elapsed, promise, cpuid, f = std::forward<F>(f),
                       args = std::make_tuple(std::forward<Args>(args)...)] {
        SetCpuAffinityForCurrentThread(cpuid);
        SetSchedPolicyForCurrentThread(kRoundRobin, MaxPriority(kRoundRobin));
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // pause 200 ms
        std::apply(
            [&promise, &f, &ms_elapsed](auto &&... args) {
              auto start = Now();
              promise->set_value_at_thread_exit(std::invoke(f, args...));
              ms_elapsed = (f64)MicrosecondsElapsed(start) / 1000;
            },
            args);
      });
      tid.detach();
      return future.get();
    }


    template <typename F, typename... Args>
    RepeatedLatencies RunRepeatedlyWithoutNoiseless(int reps, F &&f, Args &&... args)
    {
      RepeatedLatencies res;
      for (int i = 0; i < reps; i++)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        f64 ms_elapsed;
        RunWithoutNoiseless(ms_elapsed, std::forward<F>(f), std::forward<Args>(args)...);
        res.latencies.push_back(ms_elapsed);
        res.mean += ms_elapsed;
        res.min = std::min(res.min, ms_elapsed);
        res.max = std::max(res.max, ms_elapsed);
      }
      res.mean /= res.latencies.size();
      for (size_t i = 0; i < res.latencies.size(); i++)
      {
        res.stddev += std::pow(res.latencies[i] - res.mean, 2);
      }
      res.stddev = std::sqrt(res.stddev / res.latencies.size());
      return res;
    }

    template <typename F, typename... Args>
    RepeatedLatencies RunRepeatedly(int reps, F &&f, Args &&... args)
    {
      RepeatedLatencies res;
      for (int i = 0; i < reps; i++)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        f64 ms_elapsed;
        Run(ms_elapsed, std::forward<F>(f), std::forward<Args>(args)...);
        res.latencies.push_back(ms_elapsed);
        res.mean += ms_elapsed;
        res.min = std::min(res.min, ms_elapsed);
        res.max = std::max(res.max, ms_elapsed);
      }
      res.mean /= res.latencies.size();
      for (size_t i = 0; i < res.latencies.size(); i++)
      {
        res.stddev += std::pow(res.latencies[i] - res.mean, 2);
      }
      res.stddev = std::sqrt(res.stddev / res.latencies.size());
      return res;
    }

  private:
    Noiseless()
    {
      nr_cpus_ = std::thread::hardware_concurrency();
      isolated_cpu_id_ = nr_cpus_ - 1;
      std::stringstream ss;
      CPU_ZERO(&cpu_set_);
      for (int i = 0; i < nr_cpus_; i++)
      {
        CPU_SET(i, &cpu_set_);
        if(i != isolated_cpu_id_) ss<<i;
        if(i != nr_cpus_ -2) ss<<",";
      }
      other_cpu_str_ = ss.str();
      // move threads to other cpus      
      std::system(GenStr("sudo tuna --cpu=", isolated_cpu_id_, " --isolate").c_str());
      // move irqs to other cpus 
      std::system(GenStr("sudo tuna --irqs=\\* --cpu=", other_cpu_str_, " --move --spread").c_str());
    }
    ~Noiseless()
    {
      // allow other threads to run on isolated_cpu_id_
      std::system(GenStr("sudo tuna --cpu=", isolated_cpu_id_, " --include").c_str());
      // reactivate irqs on isolated_cpu_id_
      std::system(GenStr("sudo tuna --irqs=\\* --cpu=", other_cpu_str_, ",", isolated_cpu_id_, " --move --spread").c_str());
    }
    std::string other_cpu_str_;
    int isolated_cpu_id_;
    int nr_cpus_;
    cpu_set_t cpu_set_;
  };

} // namespace noiseless