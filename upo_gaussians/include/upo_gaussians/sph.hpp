#pragma once
#include "types.hpp"

namespace upo_gaussians {

	template <unsigned Level>
	static constexpr unsigned SPH_NUM_COEFS = (Level+1)*(Level+1);

	template <unsigned Level, typename Scalar>
	static inline auto make_sph(Vec<3,Scalar> const& v) -> Vec<SPH_NUM_COEFS<Level>, Scalar>
	{
		static_assert(Level <= 3, "Unimplemented SPH degree");
		Vec<SPH_NUM_COEFS<Level>, Scalar> ret;

		do {
			// l = 0
			ret( 0) = 1;

			// l = 1
			if constexpr (Level < 1) break;
			ret( 1) = v.y();
			ret( 2) = v.z();
			ret( 3) = v.x();

			// l = 2
			if constexpr (Level < 2) break;
			ret( 4) = v.x()*v.y();
			ret( 5) = v.y()*v.z();
			ret( 6) = 3*v.z()*v.z() - 1;
			ret( 7) = v.x()*v.z();
			ret( 8) = v.x()*v.x() - v.y()*v.y();

			// l = 3
			if constexpr (Level < 3) break;
			ret( 9) = v.y()*(3*v.x()*v.x() - v.y()*v.y());
			ret(10) = v.x()*v.y()*v.z();
			ret(11) = v.y()*(5*v.z()*v.z() - 1);
			ret(12) = 5*v.z()*v.z()*v.z() - 3*v.z();
			ret(13) = v.x()*(5*v.z()*v.z() - 1);
			ret(14) = (v.x()*v.x() - v.y()*v.y())*v.z();
			ret(15) = v.x()*(v.x()*v.x() - 3*v.y()*v.y());
		} while (0);

		return ret;
	}

	Vec<6> make_rcs_gradient(Vec<3> const& in_pos, Vec<3> const& in_gpos, Vec<16> const& in_rcs);

}
