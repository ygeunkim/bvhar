#ifndef BVHAR_CORE_PROGRESS_H
#define BVHAR_CORE_PROGRESS_H

// #include "./commondefs.h"
// #include "./omp.h"
#include "./common.h"

namespace bvhar {

class ProgressInterface;
class BarProgress;
class SpdlogProgress;
class bvharprogress;

class ProgressInterface {
public:
	ProgressInterface(
		int total, bool verbose,
		const std::string& prefix = "", const std::string& suffix = ""
	)
	: total(total), verbose(verbose), bar_prefix(prefix), bar_suffix(suffix) {}
	virtual ~ProgressInterface() = default;

	void setPrefix(const std::string& prefix) {
		bar_prefix = prefix;
	}

	void setSuffix(const std::string& suffix) {
		bar_suffix = suffix;
	}
	
	virtual void update(int curr_it) = 0;

	virtual void warnInterrupt(int curr_it) {}

	virtual void flush() {}

	virtual void drop() {}

protected:
	int total;
	bool verbose;
	std::string bar_prefix, bar_suffix;
};

class BarProgress : public ProgressInterface {
public:
	BarProgress(
		int total, int width, bool verbose,
		const char& default_bar = '-',
		const std::string& bar_shape = "=",
		const std::string& prefix = "",
		const std::string& suffix = ""
	)
	: ProgressInterface(total, verbose, prefix, suffix),
		width(width), percent(0),// current(0),
		progress_str(width + bar_shape.length() - 1, default_bar), bar_shape(bar_shape) {}

	virtual ~BarProgress() = default;

	void setWidth(int len) {
		width = len;
	}

	void update(int curr_it) override {
		if (!verbose) {
			return;
		}
		percent = curr_it * 100 / total;
		int curr_len = percent * width / 100;
		drawProgress(curr_len);
		std::lock_guard<std::mutex> lock(mtx);
		// BVHAR_COUT << "\r" << bar_prefix << " ["
		// 	<< progress_str << "] " << percent << "% (" << bar_suffix << ")" << std::flush;
		// // BVHAR_FLUSH;
		// if (curr_it >= total) {
		// 	BVHAR_COUT << BVHAR_ENDL;
		// }
	#ifdef _OPENMP
		#pragma omp master
	#endif
		{
			BVHAR_COUT << "\r" << bar_prefix << " ["
				<< progress_str << "] " << percent << "% (" << bar_suffix << ")" << std::flush;
			if (curr_it >= total) {
				BVHAR_COUT << BVHAR_ENDL;
			}
		}
	}

	// void flush() override {
	// 	BVHAR_FLUSH;
	// }

private:
	int width, percent;
	std::mutex mtx;
	std::string progress_str, bar_shape;

	void drawProgress(int curr_len) {
		for (int i = 0; i < curr_len; ++i) {
			progress_str.replace(i, bar_shape.length(), bar_shape);
		}
	}
};

class SpdlogProgress : public ProgressInterface {
public:
	SpdlogProgress(
		int total, int len, bool verbose,
		const std::string& prefix = "", const std::string& suffix = ""
	)
	: ProgressInterface(total, verbose, prefix, suffix),
		logging_freq(total / len) {
		logger = spdlog::get(bar_prefix);
		if (logger == nullptr) {
			logger = BVHAR_SPDLOG_SINK_MT(bar_prefix);
		}
		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
		if (logging_freq == 0) {
			logging_freq = 1;
		}
	}
	virtual ~SpdlogProgress() = default;

	void update(int curr_it) override {
		if (verbose && curr_it % logging_freq == 0) {
			logger->info("{} / {} ({})", curr_it, total, bar_suffix);
		}
	}

	void warnInterrupt(int curr_it) override {
		logger->warn("User interrupt in {} / {}", curr_it, total);
	}

	void flush() override {
		logger->flush();
	}

	void drop() override {
		spdlog::drop(bar_prefix);
	}

private:
	std::shared_ptr<spdlog::logger> logger;
	int logging_freq;
};

class bvharprogress {
public:
	bvharprogress(int total, bool verbose) : _current(0), _total(total), _width(50), _verbose(verbose) {}
	virtual ~bvharprogress() = default;
	void increment() {
		if (omp_get_thread_num() == 0) {
			_current++;
		} else {
			_current.fetch_add(1, std::memory_order_relaxed);
		}
	}
	void update() {
		if (!_verbose || omp_get_thread_num() != 0) {
			return; // not display when verbose is false
		}
		int percent = _current * 100 / _total;
		BVHAR_COUT << "\r";
		for (int i = 0; i < _width; i++) {
			if (i < (percent * _width / 100)) {
				BVHAR_COUT << "#";
			} else {
				BVHAR_COUT << " ";
			}
		}
		BVHAR_COUT << " " << percent << "%";
		BVHAR_FLUSH;
		if (_current >= _total) {
			BVHAR_COUT << BVHAR_ENDL;
		}
	}
private:
	std::atomic<int> _current;
	int _total;
	int _width;
	bool _verbose;
};

} // namespace bvhar

#endif // BVHAR_CORE_PROGRESS_H