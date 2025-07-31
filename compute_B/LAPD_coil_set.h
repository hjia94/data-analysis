// LAPD_coil_set.h

#ifndef LAPD_coil_set_h_included
#define LAPD_coil_set_included

#include <vector>
#include <iostream>
#include <iomanip>
#include "..\EllipticA.h"
#include "..\EllipticB.h"

#define cYellow 1
#define cPurple 2
#define cBlack  3

class coil_data
{ public:
	int      color;              // either cYellow or cPurple
	int      supply_number;      // power supply number, 1-based
	double   z;                  // coil mean z location in m; 0 is at far end of machine from cathode
	double   eff_port_number;    // effective port number for the coil
	double   a;                  // coil mean radius
	double   num_turns;          // number of turns
	double   current;            // current in A
	
	coil_data(int col, int s_num, double zz, double epn, double aa, double nt, double cur)
	: color(col)
	, supply_number(s_num)
	, z(zz)
	, eff_port_number(epn)
	, a(aa)
	, num_turns(nt)
	, current(cur)
	{}
	
};

// Note: on July 28, 2022 I found that a nmuber of coil z-locations were out of order. This did not 
//       matter if the two coils were on the same power supply, but in the two cases noted below
//       they were actually on the *wrong* supplies.  Check with Shreekrishna..

// I DID NOT REPAIR THE PORT NUMBERS WHEN I MOVED THE COILS AROUND

inline std::vector<coil_data> BaO_coil_set()
{	std::vector<coil_data> cd;
	//                         color    PS#    Zloc     port#  radius  #turns  current
	cd.push_back(	coil_data( cYellow,	 1,	16.93350,	 0.0,  	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 1,	16.77375,	 0.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 1,	16.45425,	 1.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 1,	16.13475,	 2.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 1,	15.81525,	 3.5, 	.67842,	10, 0.0) );
	
	cd.push_back(	coil_data( cYellow,	 2,	15.49575,	 4.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 2,	15.17625,	 5.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 2,	14.85675,	 6.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 2,	14.53725,	 7.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 2,	14.21775,	 8.5, 	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 2,	13.89825,	 9.5, 	.67842,	10, 0.0) );
	
	cd.push_back(	coil_data( cPurple,	 5,	13.62175,	10.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	13.53575,	10.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	13.30225,	11.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	13.21625,	11.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.98275,	12.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.89675,	12.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.66325,	13.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.57725,	13.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.34375,	14.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.25775,	14.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	12.02425,	15.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 5,	11.93825,	15.635,	.67348,	14, 0.0) );
	
	cd.push_back(	coil_data( cPurple,	 6,	11.70475,	16.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	11.61875,	16.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	11.38525,	17.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	11.29925,	17.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	11.06575,	18.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	10.97975,	18.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	10.74625,	19.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	10.66025,	19.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	10.42675,	20.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	10.34075,	20.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 6,	10.10725,	21.365,	.67348,	14, 0.0) );      // this coil's z location and next one were reversed - 22-07-28
	
	cd.push_back(	coil_data( cPurple,	 7,	10.02125,	21.65,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 9.78775, 	22.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,  9.70175, 	22.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 9.46825, 	23.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 9.38225, 	23.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 9.14875, 	24.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 9.06275, 	24.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 8.82925, 	25.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 8.74325, 	25.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 8.50975, 	26.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 7,	 8.42375, 	26.635,	.67348,	14, 0.0) );
	
	cd.push_back(	coil_data( cPurple,	 8,	 8.19025, 	27.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 8.10425, 	27.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 7.87075, 	28.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 7.78475, 	28.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 7.55125, 	29.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 7.46525, 	29.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 7.23175, 	30.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 7.14575, 	30.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 6.91225, 	31.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 6.82625, 	31.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 8,	 6.59275, 	32.635,	.67348,	14, 0.0) );      // this coil's z location and next one were reversed - 22-07-28
	
	cd.push_back(	coil_data( cPurple,	 9,	 6.50675, 	32.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 6.27325, 	33.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 6.18725, 	33.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 5.95375, 	34.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 5.86775, 	34.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 5.63425, 	35.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 5.54825, 	35.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 5.31475, 	36.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 5.22875, 	36.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 4.99525, 	37.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	 9,	 4.90925, 	37.635,	.67348,	14, 0.0) );
	
	cd.push_back(	coil_data( cPurple,	10,	 4.67575, 	38.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 4.58975, 	38.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 4.35625, 	39.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 4.27025, 	39.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 4.03675, 	40.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 3.95075, 	40.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 3.71725, 	41.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 3.63125, 	41.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 3.39775, 	42.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 3.31175, 	42.635,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 3.07825, 	43.365,	.67348,	14, 0.0) );
	cd.push_back(	coil_data( cPurple,	10,	 2.99225, 	43.635,	.67348,	14, 0.0) );
	
	cd.push_back(	coil_data( cYellow,	 3,	 2.71575, 	44.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 3,	 2.39625, 	45.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 3,	 2.07675, 	46.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 3,	 1.75725, 	47.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 3,	 1.43775, 	48.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 3,	 1.11825, 	49.5,	.67842,	10, 0.0) );
	
	cd.push_back(	coil_data( cYellow,	 4,	  .79875, 	50.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 4,	  .47925, 	51.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 4,	  .15975, 	52.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 4,	 -.15975, 	53.5,	.67842,	10, 0.0) );
	cd.push_back(	coil_data( cYellow,	 4,	 -.31950, 	54.0,	.67842,	10, 0.0) );

	return cd;
}

inline std::vector<coil_data> LaB6_coil_set()
{	std::vector<coil_data> cd;
	//                         color    PS#    Zloc        port#  radius  #turns  current
	cd.push_back(	coil_data( cBlack,  12,   19.12413-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   19.03475-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   18.94538-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   18.85600-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   18.76663-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   18.67725-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   18.58788-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  12,   18.49850-0.4,     -1,     0.55,    16,     0.0) );

	cd.push_back(	coil_data( cBlack,  11,   18.40913-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   18.31975-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   18.23038-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   18.14100-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   18.05163-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   17.96225-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   17.87288-0.4,     -1,     0.55,    16,     0.0) );
	cd.push_back(	coil_data( cBlack,  11,   17.78350-0.4,     -1,     0.55,    16,     0.0) );

	std::vector<coil_data> b = BaO_coil_set();
	cd.insert(cd.end(), b.begin(), b.end());   // concatenate
	return cd;
}


class LAPD_coil_set
{ public:
	std::vector<coil_data> cd;

	// constructor
	LAPD_coil_set()
	{
		cd = LaB6_coil_set();
		set_uniform_field(0.1);
	}
	
	void set_uniform_field(double B0)  // B0 in Tesla
	{	double m = B0/0.1;
		set_supply_current(2600.*m, 1, 2, 3, 4);		  // 2580    vs 2600
		set_supply_current(910.*m, 5, 6, 7, 8, 9, 10);	  // 911     vs 910
		set_supply_current(555.*m, 11, 12);               // 2.22 kA gives 4 kG according to sign on supply
	}
	
	void set_supply_current(double current, unsigned Sa, unsigned Sb = 0, unsigned Sc = 0, unsigned Sd = 0, unsigned Se = 0, unsigned Sf = 0)
	{	for(size_t i = 0;  i < cd.size();  ++i)
		{	if(cd[i].supply_number == Sa  ||
			   cd[i].supply_number == Sb  ||
			   cd[i].supply_number == Sc  ||
			   cd[i].supply_number == Sd  ||
			   cd[i].supply_number == Se  ||
			   cd[i].supply_number == Sf    )
				cd[i].current = current;
		}
	}

	void set_supply_currents(double i1, double i2, double i3, double i4, double i5, double i6, double i7, double i8, double i9, double i10, double i11, double i12)
	{	set_supply_current( i1,  1);
		set_supply_current( i2,  2);
		set_supply_current( i3,  3);
		set_supply_current( i4,  4);
		set_supply_current( i5,  5);
		set_supply_current( i6,  6);
		set_supply_current( i7,  7);
		set_supply_current( i8,  8);
		set_supply_current( i9,  9);
		set_supply_current(i10, 10);
		set_supply_current(i11, 11);
		set_supply_current(i12, 12);
	}

	double z_to_eff_port_number(double z) const
	{	if(z < cd.back().z)
			return cd.back().eff_port_number;
		if(z > cd[0].z)
			return cd[0].eff_port_number;

		for(size_t i = 0;  i < cd.size()-1;  ++i)
		{	if(cd[i].z >= z  &&  cd[i+1].z < z)
			{	double f = (z - cd[i+1].z) / (cd[i].z - cd[i+1].z);
				double epn = cd[i+1].eff_port_number + f * (cd[i].eff_port_number - cd[i+1].eff_port_number);
				return epn;
			}
		}

		return 0.;
	}


	// obs.x, obs.y, obs.z = location of observer (compute field here)
	// z = 0 is at end of machine; z is positive toward the cathode
	// If observer is past far end of machine, we expect obs.z < 0 for the coordinates as defined here
	//
	CartB compute_B(CartPt const& obs) const
	{
		CylPt cyl_o(obs);
		CylB sum;
		
		for(size_t i = 0;  i < cd.size();  ++i)
		{
			double Zc = cd[i].z;
			double ac = cd[i].a;

			double R = cyl_o.R / ac;
			double Bnorm = mu0 * cd[i].current * cd[i].num_turns / (2 * ac);

			double z = (obs.z - Zc) / ac;				// when past far end of machine, we expect z/ac < 0 - i.e. observer is "below" the plane of the coil

			sum += Bnorm * EllipticB(R, z);
		}

		return CartB(sum, cyl_o);
	}

	coil_data operator[](unsigned i) const
	{	return cd[i];
	}


	double compute_Aphi(CylPt const& obs) const
	{
		double sum = 0;
		for(size_t i = 0;  i < cd.size();  ++i)
		{
			double Zc = cd[i].z;
			double ac = cd[i].a;

			double R = obs.R / ac;
			double z = (obs.z - Zc) / ac;

			sum += mu0 * cd[i].current * cd[i].num_turns * EllipticA(R, z);
		}
		
		return sum;
	}

	double Psi(CylPt const& obs) const  // = 2*pi* R * Aphi(R,z)
	{	double const pi = acos(-1.);
		return 2 * pi * obs.R * compute_Aphi(obs);
//		return compute_Aphi(obs);
	}

	friend std::ostream& operator<<(std::ostream& os, LAPD_coil_set const& cs)
	{
		os << "       z       a_eff      #turns    cur     supply#" << std::endl;
		for (std::vector<coil_data>::const_iterator it = cs.cd.begin(); it != cs.cd.end(); ++it)
		{
			os << std::setw(10) << std::setprecision(5) << it->z;
			os << std::setw(10) << std::setprecision(3) << it->a;
			os << std::setw(10) << it->num_turns;
			os << std::setw(10) << std::setprecision(4) << it->current;
			os << std::setw(10) << std::setprecision(3) << it->supply_number;
			os << std::endl;
		}
		return os;
	}

};




#endif // ndef LAPD_coil_set_h_included
