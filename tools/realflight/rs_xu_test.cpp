// Isolate the D455 auto-exposure XU failure: try the same option calls
// librealsense uses when setting exposure/gain, with the stream off and on.
#include <iostream>
#include <chrono>
#include <string>
#include <thread>

#include <librealsense2/rs.hpp>

static void try_ae(rs2::sensor& s, const std::string& tag) {
    try {
        bool supports = s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE);
        std::cout << "[" << tag << "] supports ENABLE_AUTO_EXPOSURE=" << supports << std::endl;
        if (!supports) return;
        float v = s.get_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE);
        std::cout << "[" << tag << "] auto_exposure=" << v << std::endl;
        s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.f);
        std::cout << "[" << tag << "] set auto_exposure=0 OK" << std::endl;
        s.set_option(RS2_OPTION_EXPOSURE, 10000.f);
        std::cout << "[" << tag << "] set exposure=10000 OK, now="
                  << s.get_option(RS2_OPTION_EXPOSURE) << std::endl;
        s.set_option(RS2_OPTION_GAIN, 32.f);
        std::cout << "[" << tag << "] set gain=32 OK, now="
                  << s.get_option(RS2_OPTION_GAIN) << std::endl;
    } catch (const rs2::error& e) {
        std::cout << "[" << tag << "] RS2_ERROR: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[" << tag << "] EXCEPTION: " << e.what() << std::endl;
    }
}

static void probe_ae_range(rs2::sensor& s, const std::string& tag) {
    try {
        auto r = s.get_option_range(RS2_OPTION_ENABLE_AUTO_EXPOSURE);
        std::cout << "[" << tag << "] ae range min=" << r.min << " max=" << r.max
                  << " step=" << r.step << " def=" << r.def << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[" << tag << "] ae range ERROR: " << e.what() << std::endl;
    }
}

static void try_ae_cycle(rs2::sensor& s, const std::string& tag) {
    for (int i = 1; i <= 3; ++i) {
        try {
            float v = s.get_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE);
            std::cout << "[" << tag << "] iter" << i << " ae=" << v << std::endl;
            s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);
            std::cout << "[" << tag << "] iter" << i << " ae->1 OK" << std::endl;
            s.set_option(RS2_OPTION_EXPOSURE, 10000.f);
            std::cout << "[" << tag << "] iter" << i << " exposure set OK, now="
                      << s.get_option(RS2_OPTION_EXPOSURE) << std::endl;
            s.set_option(RS2_OPTION_GAIN, 32.f);
            std::cout << "[" << tag << "] iter" << i << " gain set OK, now="
                      << s.get_option(RS2_OPTION_GAIN) << std::endl;
        } catch (const rs2::error& e) {
            std::cout << "[" << tag << "] iter" << i << " RS2_ERROR: " << e.what()
                      << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[" << tag << "] iter" << i << " EXCEPTION: " << e.what()
                      << std::endl;
        }
    }
}

static void sensor_stream_and_test(rs2::sensor& depth, const std::string& tag,
                                   int w, int h, int fps) {
    try {
        rs2::stream_profile chosen;
        for (auto& p : depth.get_stream_profiles()) {
            if (p.stream_type() != RS2_STREAM_DEPTH ||
                p.format() != RS2_FORMAT_Z16) continue;
            auto vp = p.as<rs2::video_stream_profile>();
            if (vp.width() == w && vp.height() == h && vp.fps() == fps) {
                chosen = p;
                break;
            }
        }
        if (!chosen) {
            std::cout << "[" << tag << "] no matching profile" << std::endl;
            return;
        }
        depth.open(chosen);
        depth.start([](rs2::frame) {});
        std::cout << "[" << tag << "] sensor stream " << w << "x" << h << "@"
                  << fps << " started" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        probe_ae_range(depth, tag);
        try_ae_cycle(depth, tag);
        depth.stop();
        depth.close();
    } catch (const std::exception& e) {
        std::cout << "[" << tag << "] STREAM_ERROR: " << e.what() << std::endl;
    }
}

int main() {
    try {
        rs2::context ctx;
        auto devs = ctx.query_devices();
        if (devs.size() == 0) {
            std::cerr << "no device" << std::endl;
            return 2;
        }
        auto dev = devs[0];
        std::cout << "device: " << dev.get_info(RS2_CAMERA_INFO_NAME)
                  << " sn=" << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
                  << " fw=" << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION)
                  << std::endl;

        rs2::sensor depth;
        for (auto& s : dev.query_sensors()) {
            std::string name = s.get_info(RS2_CAMERA_INFO_NAME);
            std::cout << "sensor: " << name << std::endl;
            if (name.find("Stereo") != std::string::npos ||
                name.find("Depth") != std::string::npos) {
                depth = s;
            }
        }
        if (!depth) {
            std::cerr << "no depth sensor" << std::endl;
            return 3;
        }

        probe_ae_range(depth, "no-stream");
        try_ae(depth, "no-stream");
        probe_ae_range(depth, "no-stream");
        sensor_stream_and_test(depth, "stream-424x240", 424, 240, 15);
        sensor_stream_and_test(depth, "stream-640x480", 640, 480, 15);
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
