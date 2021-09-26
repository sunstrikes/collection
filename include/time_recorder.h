//
// Created by Sun,Minqi on 2021/9/17.
//

#ifndef TEST_TIMERECORDER_H
#define TEST_TIMERECORDER_H
#include <string>

template <class Duration>
std::string human_readable_time(Duration cost) {
    using namespace std::chrono_literals;
    auto abs_cost = cost.count() < 0 ? -cost : cost;
    if (abs_cost < 1ms) {
        return std::to_string(
                std::chrono::duration_cast<std::chrono::duration<double, std::chrono::microseconds ::period>>(cost).count())
               + "us";
    } else if (abs_cost < 1s) {
        return std::to_string(
                std::chrono::duration_cast<std::chrono::duration<double, std::chrono::milliseconds::period>>(cost).count())
               + "ms";
    } else if (abs_cost < 1min) {
        return std::to_string(
                std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(cost).count())
               + "s";
    } else if (abs_cost < 1h) {
        return std::to_string(
                std::chrono::duration_cast<std::chrono::duration<double, std::chrono::minutes::period>>(cost).count())
               + "min";
    } else if (abs_cost < std::chrono::days(1)) {
        return std::to_string(
                std::chrono::duration_cast<std::chrono::duration<double, std::chrono::hours::period>>(cost).count())
               + "h";
    }
    return std::to_string(
            std::chrono::duration_cast<std::chrono::duration<double, std::chrono::days::period>>(cost).count())
           + "day";
}

class TimeRecorder {
public:
    TimeRecorder(std::string name);
    ~TimeRecorder();
private:
    std::string _name;
    std::chrono::steady_clock::time_point _start_time;
};


#endif //TEST_TIMERECORDER_H
