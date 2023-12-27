#ifndef BVHARINTERRUPT_H
#define BVHARINTERRUPT_H

#include <csignal>
#include <atomic>

class bvharinterrupt {
private:
	static std::atomic<bool> _interrupted;
	static void handle_signal(int signal);
public:
	bvharinterrupt();
	virtual ~bvharinterrupt() = default;
	static bool is_interrupted();
};

#endif
