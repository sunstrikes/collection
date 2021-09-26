//
// Created by Sun,Minqi on 2021/9/17.
//

#include "time_recorder.h"
#include <glog/logging.h>
#include <utility>

TimeRecorder::TimeRecorder(std::string name)
        : _name(std::move(name)), _start_time(std::chrono::steady_clock::now()) {
    LOG(INFO) << _name << " @ begin";
}
TimeRecorder::~TimeRecorder() {
    using namespace std::chrono_literals;
    auto cost = std::chrono::steady_clock::now() - _start_time;
    LOG(INFO) << _name << " @ cost " << human_readable_time(cost);
}