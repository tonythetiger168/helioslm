// SPDX-License-Identifier: Apache-2.0
// Connect the VIP's 32-bit address bus to the Open IP wrapper without aliasing.
module apb_portfolio_bridge (
    input logic clk, rst_n, select, enable, write,
    input logic [31:0] address, write_data,
    input logic [3:0] strobe,
    output wire [31:0] read_data,
    output wire ready, error
);
    wire mapped = address[31:12] == 20'b0;
    wire [31:0] dut_data;
    wire dut_ready, dut_error;

    APB_apb dut (
        .PCLK(clk), .PRESETN(rst_n), .PSEL(select && mapped),
        .PENABLE(enable), .PWRITE(write), .PADDR(address[11:0]),
        .PWDATA(write_data), .PSTRB(strobe), .PRDATA(dut_data),
        .PREADY(dut_ready), .PSLVERR(dut_error), .irq(),
        .psel(1'b0), .penable(1'b0), .paddr(32'b0), .pwrite(1'b0),
        .pwdata(32'b0), .prdata(), .pready(), .pslverr()
    );

    assign ready = mapped ? dut_ready : select && enable;
    assign error = mapped ? dut_error : select && enable;
    assign read_data = mapped ? dut_data : 32'b0;
endmodule
