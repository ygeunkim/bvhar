#ifndef BVHARINTERRUPT_H
#define BVHARINTERRUPT_H

#include <csignal>
#include <atomic>

namespace bvhar {

class bvharinterrupt {
private:
	static std::atomic<bool> _interrupted;
	static void handle_signal(int signal);
public:
	bvharinterrupt();
	virtual ~bvharinterrupt() = default;
	static bool is_interrupted();
	static void reset();
};

} // namespace bvhar

#endif // BVHARINTERRUPT_H