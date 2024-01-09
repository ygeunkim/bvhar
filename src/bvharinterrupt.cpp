#include "bvharinterrupt.h"

std::atomic<bool> bvharinterrupt::_interrupted(false);

bvharinterrupt::bvharinterrupt() {
	std::signal(SIGINT, bvharinterrupt::handle_signal);
}

bool bvharinterrupt::is_interrupted() {
	return _interrupted.load();
}

void bvharinterrupt::handle_signal(int signal) {
	if (signal == SIGINT) {
		_interrupted.store(true);
	}
}
