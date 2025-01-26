#ifndef BVHAR_CORE_INTERRUPT_H
#define BVHAR_CORE_INTERRUPT_H

#include <csignal>
#include <atomic>

namespace bvhar {

class bvharinterrupt;

class bvharinterrupt {
private:
	static std::atomic<bool>& interrupted() {
		static std::atomic<bool> _interrupted(false);
		return _interrupted;
	}
	static void handle_signal(int signal) {
		if (signal == SIGINT) {
			interrupted().store(true);
		}
	}
public:
	bvharinterrupt() {
		reset();
		std::signal(SIGINT, bvharinterrupt::handle_signal);
	}
	virtual ~bvharinterrupt() = default;
	static bool is_interrupted() {
		return interrupted().load();
	}
	static void reset() {
		interrupted().store(false);
	}
};

} // namespace bvhar

#endif // BVHAR_CORE_INTERRUPT_H