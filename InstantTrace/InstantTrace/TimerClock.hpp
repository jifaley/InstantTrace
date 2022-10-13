#pragma once
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

class TimerClock
{
public:
	TimerClock()
	{
		update();
	}

	~TimerClock()
	{
	}

	void update()
	{
		_start = high_resolution_clock::now();
	}
	//��ȡ��
	double getTimerSecond()
	{
		return getTimerMicroSec() * 0.000001;
	}
	//��ȡ����
	double getTimerMilliSec()
	{
		return getTimerMicroSec()*0.001;
	}
	//��ȡ΢��
	long long getTimerMicroSec()
	{
		//��ǰʱ�Ӽ�ȥ��ʼʱ�ӵ�count
		return duration_cast<microseconds>(high_resolution_clock::now() - _start).count();
	}
private:
	time_point<high_resolution_clock>_start;
};