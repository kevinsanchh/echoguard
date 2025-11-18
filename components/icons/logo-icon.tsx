import * as React from "react";

interface IconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
}

export function LogoIcon({ size = 24, ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path
        d="M20,13C20,18 16.5,20.5 12.34,21.95C12.122,22.024 11.885,22.02 11.67,21.94C7.5,20.5 4,18 4,13L4,6C4,5.451 4.451,5 5,5C7,5 9.5,3.8 11.24,2.28C11.676,1.908 12.324,1.908 12.76,2.28C14.51,3.81 17,5 19,5C19.549,5 20,5.451 20,6L20,13Z"
        stroke="currentColor"
        strokeWidth={2}
        fill="none"
        fillRule="nonzero"
      />
      <g transform="matrix(0.5,0,0,0.5,11.733333,12.266667)">
        <g transform="matrix(1,0,0,1,-12,-12)">
          <path
            d="M2,10L2,13"
            stroke="currentColor"
            strokeWidth={2}
            fill="none"
            fillRule="nonzero"
          />
          <path
            d="M6,6L6,17"
            stroke="currentColor"
            strokeWidth={2}
            fill="none"
            fillRule="nonzero"
          />
          <path
            d="M10,3L10,21"
            stroke="currentColor"
            strokeWidth={2}
            fill="none"
            fillRule="nonzero"
          />
          <path
            d="M14,8L14,15"
            stroke="currentColor"
            strokeWidth={2}
            fill="none"
            fillRule="nonzero"
          />
          <path
            d="M18,5L18,18"
            stroke="currentColor"
            strokeWidth={2}
            fill="none"
            fillRule="nonzero"
          />
          <path
            d="M22,10L22,13"
            stroke="currentColor"
            strokeWidth={2}
            fill="none"
            fillRule="nonzero"
          />
        </g>
      </g>
    </svg>
  );
}
